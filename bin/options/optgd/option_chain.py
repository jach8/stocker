

import sys
import pandas as pd 
import numpy as np 
import yfinance as yf 
import datetime as dt 

sys.path.append('/Users/jerald/Documents/Dir/Python/Stocks')
from bin.options.optgd.db_connect import Connector
from models.bsm.bs2 import bs_df as new_bsdf
from models.bsm.bsModel import bs_df


class OptionChain(Connector):
    def __init__(self, connections):
        """
        Module for obtaining the option chain from Yahoo Finance. 
        
        Args: 
            connections: Dictionary of the connections.
        
        Methods:
            get_option_chain(stock:str) -> pd.DataFrame: Get the option chain for a stock.
            _check_for_stock_in_option_db(stock:str) -> bool: Check if the stock is in the option database.
            insert_new_chain(stock:str) -> pd.DataFrame: Insert a new chain into the database.
        
        """
        super().__init__(connections)
        
    def get_option_chain(self, stock:str) -> pd.DataFrame:
        """ 
        Gets the option chain from Yahoo Finance for a stock. 
        
        Args: 
            stock (str): Stock symbol.
        
        Returns:
            pd.DataFrame: Option Chain DataFrame.
        
        
        """
        try:
            symbol = stock
            tk = yf.Ticker(symbol)
            last = tk.history(period = '1d').iloc[-1]["Close"]
            
            exps = tk.options
            option_list = []
            for exp in exps: 
                chain = tk.option_chain(exp)
                puts, calls = chain.puts, chain.calls
                puts['type'] = 'Put'
                calls['type'] = 'Call'
                opt = pd.concat([puts, calls])
                opt['expiry'] = pd.to_datetime(exp)
                option_list.append(opt)
                
            options = pd.concat(option_list)
            if len(options) == 0:
                return None
            else:
                options['mid'] = (options['bid'] + options['ask']) / 2
                options['cash'] = abs((last - options['strike'])* options['openInterest'])
                options['stk_price'] = round(last, 2)
                options = options.drop(columns = ['contractSize', 'currency'])
                options.insert(0, 'gatherdate', dt.datetime.today().strftime('%Y-%m-%d %H:%M:%S'))
                options['timeValue'] = (pd.to_datetime(options['expiry']) - pd.to_datetime(options['gatherdate']))/ np.timedelta64(1,'D')/252
                options.columns = [x.lower() for x in options.columns]
                options.expiry = pd.to_datetime(options.expiry)
                # options = bs_df(options)
                return options
        except Exception as e:
            print(e)
            return None
        
    def _check_for_stock_in_option_db(self, stock:str) -> bool:
        """ 
        Check if the stock is in the option database. 
        Args:
            stock (str): Stock Symbol
        
        Returns:
            bool: True if the stock is in the database, False if not.
        
        """
        cursor = self.option_db.cursor()
        query = f"""
        select exists(select 1 from sqlite_master where type='table' and name='{stock}')
        """
        valid = cursor.execute(query).fetchone()[0]
        return bool(valid)
        
            
    def insert_new_chain(self, stock: str) -> pd.DataFrame:
        """
        Insert a new chain into the database. If the stock is in the database, append the new chain.
        Otherwise the chain will be replaced and added to the database. 
        
        Args:
            stock (str): Stock Symbol
        
        Returns:
            pd.DataFrame: Option Chain DataFrame.    
    
        """
        df = self.get_option_chain(stock)
        if df is None:
            return None
        else:
            if len(df)> 0:
                if self._check_for_stock_in_option_db(stock) == True:
                    df.to_sql(stock, self.write_option_db, if_exists = 'append', index = False)
                else:
                    df.to_sql(stock, self.write_option_db, if_exists = 'replace', index = False)
                self.write_option_db.commit()
            return df
        
    
if __name__ == "__main__":
    import time
    from tqdm import tqdm
    from bin.main import Manager, get_path
    connections = get_path()
    start_time = time.time()
    oc = OptionChain(connections)
    oc.insert_new_chain('amd')
    end_time = time.time()
    print(f'\n\nTime: {end_time - start_time}')
    
    