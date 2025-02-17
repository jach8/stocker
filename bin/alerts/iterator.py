import sys
from pathlib import Path    
sys.path.append(str(Path(__file__).resolve().parents[2]))
from bin.options.optgd.db_connect import Connector as Manager


import pandas as pd
from tqdm import tqdm


class Iterator(Manager):
    def __init__(self, connections):
        super().__init__(connections)
        self.stock_dict = self.stocks.copy()
        
    def get_stocks(self, group = 'all_stocks'):
        return self.stock_dict[group]
        
    def _iterate_function(self, func, group = 'all_stocks'):
        stocks = self.get_stocks(group)
        pbar = tqdm(stocks, desc = 'Iterating')
        return [func(x) for x in pbar]
    
    def dataframe_iterator_function(self, func, group = 'all_stocks'):
        """
        Applies a given function to a group of dataframes and concatenates the results.

        Parameters:
        func (callable): The function to apply to each dataframe.
        group (str): The group of dataframes to apply the function to. Defaults to 'all_stocks'.

        Returns:
        pd.DataFrame: A concatenated dataframe resulting from applying the function to each dataframe in the group.
        """
        lodf = self._iterate_function(func, group = group)
        return pd.concat(lodf)
    
    def query_iteroator(self, query, connection, group = 'all_stocks'):
        """ query must be a function that intakes one parameter: a stock """
        stocks = self.get_stocks(group)
        pbar =  tqdm(stocks, desc = 'Iterating')
        out = []
        for stock in pbar:
            q = connection.cursor()
            g = q.execute(query(stock))
            gr = g.fetchall()
            df = pd.DataFrame(gr, columns = [x[0] for x in g.description])
            out.append(df)
        return pd.concat(out)


if __name__ == '__main__':
    
    from bin.main import Manager, get_path
    
    m = get_path()
    it = Iterator(m)
    
    def test_func(stock, conn = it.vol_db):
        """ Return todays option statistics """
        df = pd.read_sql('select * from {} order by date(gatherdate) desc limit 1'.format(stock), con = conn)
        df.insert(0, 'stock', stock)
        return df
    
    # out = it._iterate_function(test_func, group = 'etf')
    
    # out = it.dataframe_iterator_function(test_func, group = 'all_stocks')
    # print(out)
    

    def test_query(stock):
        ts = f'''
        select * from "{stock}" 
        where datetime(gatherdate) = (select max(datetime(gatherdate)) from "{stock}") 
        and volume > 1000 
        and oi_chg > 0
        and impliedvolatility < iv_avg_30d
        '''
        return ts


    df = it.query_iteroator(test_query, it.change_db)
    print(df)
        