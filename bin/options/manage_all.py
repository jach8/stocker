"""
Manager for the options data pipeline. 
    1. Get new option chain data
    2. Append data to the option database. 
    3. Update the vol.db after calculating the change variables. 

"""
import sys
sys.path.append('/Users/jerald/Documents/Dir/Python/stocker')
import pandas as pd 
import numpy as np 
# import yfinance as yf 
import datetime as dt 
from tqdm import tqdm
import time 
import sqlite3 as sql

from bin.options.stat.em import Exp
from bin.options.optgd.option_chain import OptionChain
from bin.options.stat.manage_stats import Stats
# from bin.options.track.manage_tracking import Screener
from bin.options.bsm.bs2 import bs_df 


class Manager(OptionChain, Stats):
    def __init__(self, connections):
        super().__init__(connections)
        self.connections = connections
        #self.stock_list = ['spy', 'qqq' , 'iwm']
        
        
    def update_options(self):
        for stock in tqdm(self.stocks['all_stocks']):
            new_chain = self.insert_new_chain(stock)
            if new_chain is None:
                continue
            else:
                self.update_stats(stock, new_chain)
                self.run_screener(stock)
                time.sleep(5)  
    
    def get_em(self):
        return pd.concat([self._em(stock) for stock in tqdm(self.stocks['all_stocks'])])
    
    def _manage_db(self):
        """ Manage the Option Database. 
                : Purge Inactive Contracts 
        """
        for stock in tqdm(self.stocks['all_stocks']):
            self._purge_inactive(stock)
    
    
    def _test_import(self):
        print('Imported')  
        print(self.__dict__) 
        
        
    def _join_purged_data(self, stock):
        q = f'''
        select 
        *
        from {stock}
        where date(expiry) > date('now')
        '''
        # df = pd.read_sql_query(q, self.inactive_db, parse_dates = ['gatherdate']) 
        cursor = self.inactive_db.cursor()
        cursor.execute(q)
        df = pd.DataFrame(cursor.fetchall(), columns = [desc[0] for desc in cursor.description])

    def _parse_change_db(self, stock, today = True, bsdf = True):
        stock = stock.lower()
        if today: 
            q = f'''select * from {stock} where date(gatherdate) = (select max(date(gatherdate)) from {stock}) '''
        else:
            q = f'''select * from {stock}'''
        # q = f'''select * from {stock}'''
        df = pd.read_sql(q, self.change_db, parse_dates=['gatherdate'])
        df['desc'] = [self.describe_option(x) for x in df.contractsymbol]
        names = ['stock', 'type', 'strike', 'expiry']
        df[names] = pd.DataFrame(df['desc'].tolist(), index=df.index)
        df.drop(['desc'], axis=1, inplace=True)
        df = df.set_index(names).reset_index()
        df['expiry'] = pd.to_datetime(df.expiry)
        df.type = df.type.map({'C': 'Call', 'P': 'Put'})
        df['stock'] = [x.replace('$', '') for x in df.stock]
        if bsdf == True:
            days = ((df.expiry+ pd.Timedelta('16:59:59')) - df.gatherdate).dt.days 
            # if days < 1, add 1 to days
            days = days.apply(lambda x: x if x > 0 else x + 0.5)
            df['timevalue'] = days / 252
            df['cash'] = np.abs(df.strike - df.stk_price) * df.openinterest * 100 
        
            return bs_df(df)
        else: 
            return df 
    
        
    def _contract_lookup(self, stock, args):
        if 'contractsymbol' in args:
            q = f'''select * from "{stock}" where contractsymbol = "{args['contractsymbol']}"'''
        if 'strike' in args:
            if 'type' in args:
                if 'expiry' in args:
                    q = f'''select * from "{stock}" where strike = {args['strike']} and type = "{args['type']}" and expiry = "{args['expiry']}"'''
        cursor = self.option_db.cursor()
        cursor.execute(q)
        df = pd.DataFrame(cursor.fetchall(), columns = [x[0] for x in cursor.description])
        return bs_df(df)
    
    def option_custom_q(self, q, db = 'option_db'):
        cursor = sql.connect(self.connections[db]).cursor()
        cursor.execute(q)
        df = pd.DataFrame(cursor.fetchall(), columns = [x[0] for x in cursor.description])
        if db != 'change_db':
            return df
        else:
            return self.parse_change_db(df)
    
    def parse_change_db(self, df):
        """ Parse the output from the change_db or any dataframe where contractsymbol is in the dataframe

            args: 
                -df : pd.DataFrame containing contractsymbol in the columns 
        
        """
        out_desc = {x: self.describe_option(x) for x in df.contractsymbol.to_list()}
        out_desc_df = pd.DataFrame(out_desc).T.reset_index()
        out_desc_df.columns = ['contractsymbol', 'stock', 'type', 'strike', 'expiry']
        out_desc_df['stock'] = out_desc_df.stock.apply(lambda x: x.replace('$', ''))
        out_df = out_desc_df.merge(df, on = 'contractsymbol')
        out_df.strike = out_df.strike.astype(float)
        out_df.expiry = pd.to_datetime(out_df.expiry)
        return out_df
    
    
    def pcdb(self, df):
        """ Parse the output from the change_db or any dataframe where contractsymbol is in the dataframe

            args: 
                -df : pd.DataFrame containing contractsymbol in the columns 
        
        """
        out_desc = {x: self.describe_option(x) for x in df.contractsymbol.to_list()}
        out_desc_df = pd.DataFrame(out_desc).T.reset_index()
        out_desc_df.columns = ['contractsymbol', 'stock', 'type', 'strike', 'expiry']
        out_desc_df['stock'] = out_desc_df.stock.apply(lambda x: x.replace('$', ''))
        out_desc_df.expiry = pd.to_datetime(out_desc_df.expiry)
        out_desc_df.type = out_desc_df.type.map({'C': 'Call', 'P': 'Put'})
        out = out_desc_df.merge(df.reset_index(drop = True), on = 'contractsymbol')
        out.strike = out.strike.astype(float)
        return out
    
        
        
        

            
            
            
if __name__ == "__main__":
    print("You Cant go back and change the begining, but you can start right now and change the ending.")
    
    connections = {
            'inactive_db': 'bin/pipe/log/inactive.db',
            'backup_db': 'bin/pipe/log/backup.db',
            'tracking_values_db': 'data/options/bin/pipe/test_data/tracking_values.db',
            'tracking_db': 'data/options/bin/pipe/test_data/tracking.db',
            'stats_db': 'bin/pipe/test_data/stats.db',
            'vol_db': 'bin/pipe/test_data/vol.db',
            'change_db': 'bin//pipe/test_data/option_change.db', 
            'option_db': 'bin/pipe/test_data/test.db', 
            'testing_option_db': 'bin/pipe/test_data/test.db',
            'options_stat': 'bin/pipe/test_data/options_stat.db',
            'ticker_path': 'data/stocks/tickers.json'
            }

    import sys
    sys.path.append('/Users/jerald/Documents/Dir/Python/stocker')
    from bin.main import get_path
    
    connections = get_path()
    oc = Manager(connections)
    
    # print(oc.option_custom_q('select * from aapl', db = 'change_db'))
    oc._init_em_tables()
    
    
    
    