"""
Uses Yahoo Finance API to obtain option chain data for a stock. 
    1. Returns the option chain data. 

"""

import sys
import pandas as pd 
import numpy as np 
import yfinance as yf 
import datetime as dt 

sys.path.append('/Users/jerald/Documents/Dir/Python/stocker')
from bin.options.optgd.db_connect import Connector
from bin.options.bsm.bs2 import bs_df as new_bsdf
from bin.options.bsm.bsModel import bs_df


class OptionChain(Connector):
    def __init__(self, connections):
        super().__init__(connections)
        
    def get_option_chain(self, stock):
        """ Gets the option chain from Yahoo Finance for a stock. """
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
    
    def insert_new_chain(self, stock):
        """ Get New Option Chain and Update the database. """
        df = self.get_option_chain(stock)
        if df is None:
            return None
        else:
            if len(df)> 0:
                if self._check_for_stock_in_option_db(stock) == True:
                    df.to_sql(stock, self.write_option_db, if_exists = 'replace', index = False)
                else:
                    df.to_sql(stock, self.write_option_db, if_exists = 'append', index = False)
                self.write_option_db.commit()
            return df
        
    def new_bsdf(self, stock):
        """ Update the option chain with the new black scholes model """
        df = pd.read_sql(f"SELECT * FROM '{stock}' where date(gatherdate) >= date('2024-01-01')", self.option_db)
        bsdf = new_bsdf(df)
        # bsdf.to_sql(stock, self.write_option_db, if_exists = 'replace', index = False)
        return bsdf
    
    def new_bsdf_df(self, df):
        return new_bsdf(df)
    
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
        out_df.expiry = pd.to_datetime(out_df.expiry)
        return out_df
    
    
    def today_option_chain(self, stock, bsdf = True):
        """ Get the option chain for today. """
        cursor = self.option_db.cursor()
        lod = cursor.execute(f"SELECT * FROM '{stock}' where date(gatherdate) = (select max(date(gatherdate)) from {stock})").fetchall()
        df = pd.DataFrame(lod, columns = [x[0] for x in cursor.description])
        df.gatherdate = pd.to_datetime(df.gatherdate)
        df.expiry = pd.to_datetime(df.expiry)
        df = df.drop(columns = ['lasttradedate'])
        df['moneyness'] = df.strike / df.stk_price
        if bsdf == True:
            return new_bsdf(df)
        else:
            return df
     
    def today_option_chain_cdb(self, stock):
        """ Get the option chain for today. """
        cursor = self.change_db.cursor()
        lod = cursor.execute(f"SELECT * FROM '{stock}' where date(gatherdate) = (select max(date(gatherdate)) from '{stock}')").fetchall()
        df = pd.DataFrame(lod, columns = [x[0] for x in cursor.description])
        df.gatherdate = pd.to_datetime(df.gatherdate)
        df = self.parse_change_db(df)
        df.expiry = pd.to_datetime(df.expiry)
        df['moneyness'] = df.strike / df.stk_price
        return df
     
if __name__ == "__main__":
    import time
    from tqdm import tqdm

    connections = {
                ##### Price Report ###########################
                'daily_db': 'data/prices/stocks.db', 
                'intraday_db': 'data/prices/stocks_intraday.db',
                'ticker_path': 'data/stocks/tickers.json',
                ################################################
                'inactive_db': 'data/options/log/inactive.db',
                'backup_db': 'data/options/log/backup.db',
                'tracking_values_db': 'data/options/tracking_values.db',
                'tracking_db': 'data/options/tracking.db',
                'stats_db': 'data/options/stats.db',
                'vol_db': 'data/options/vol.db',
                'change_db': 'data/options/option_change.db', 
                'option_db': 'data/options/options.db', 
                'options_stat': 'data/options/options_stat.db',
                'ticker_path': 'data/stocks/tickers.json'
    }

    start_time = time.time()
    oc = OptionChain(connections)
    lodf =  [print(oc.new_bsdf(x)) for x in tqdm(oc.stocks['all_stocks'][:2])]
    end_time = time.time()
    print(f'\n\nTime: {end_time - start_time}')
    
    