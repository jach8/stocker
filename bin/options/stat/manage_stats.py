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
import yfinance as yf 
import datetime as dt 
from tqdm import tqdm
import time 

from bin.options.stat.em import Exp
from bin.options.stat.change_vars import ChangeVars
from bin.options.stat.cp import CP


class Stats(Exp, ChangeVars, CP):
    def __init__(self, connections):
        super().__init__(connections)
        
    def update_stats(self, stock, new_chain):
        self._update_change_vars(stock)
        self.update_cp(stock, new_chain)
        self._em(stock, new_chain)
        
    def _init_change_db(self):
        for stock in tqdm(self.stocks['all_stocks']):
            self._initialize_change_db(stock)
            
    def _init_vol_db(self):
        for stock in tqdm(self.stocks['all_stocks']):
            self._initialize_vol_db(stock)
            
    def clear_tables(self):
        ''' Save a log and clear All tables in the stats db'''
        # save backup file 
        self.stats_db.backup(self.backup)
        c = self.stats_db.cursor()
        tables = [i[0] for i in c.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]
        for i in tables:
            c.execute(f'drop table {i}')
            self.stats_db.commit()
        return None
    
    def cp_query(self, stock, n = 30):
        try:
            old_df = self.get_cp_from_purged_db(stock, n = n)
        except:
            old_df = pd.DataFrame()
        current_df = self._calculation(self._cp(stock, n = n))
        new_df = pd.concat([old_df, current_df], axis = 0).reset_index()
        return new_df
    
    def _init_em_tables(self):
        """ Run the intialization of the expected moves tables """
        try:
            self._initialize_em_tables()
        except:
            pass
        
    def reverse_dict(self):
        """Reverses the keys and values of a dictionary containing string keys and list of string values.

        Args:
            d: The input dictionary.

        Returns:
            A new dictionary with the reversed keys and values.
        """
        d = self.stocks.copy()
        if 'all_stocks' in d:
            del d['all_stocks']
            
        reversed_dict = {}
        for key, values in d.items():
            for value in values:
                if value not in reversed_dict:
                    reversed_dict[value] = []
                reversed_dict[value].append(key)
        
        sg = {}
        for stockname, groups in reversed_dict.items():
            if len(groups) == 1: 
                sg[stockname] = groups[0]
            else:
                sg[stockname] = groups[1]
        return sg    
    
    def _mdf(self, stock):
        df = pd.read_sql(f'''select * from "{stock}" order by datetime(gatherdate) asc''', self.vol_db, parse_dates = ['gatherdate'])
        df.gatherdate = pd.to_datetime(df.gatherdate, format = '%Y-%m-%dT%H:%M:%S')
        df.insert(0, 'stock', stock)
        df.total_oi = df.total_oi.ffill()
        df.call_oi = df.call_oi.ffill()
        df.put_oi = df.put_oi.ffill()
        df =df.round(4)

        ### Calculate Moving Averages 
        sma_vol = df['total_vol'].rolling(30).mean()
        std_vol = df['total_vol'].rolling(30).std()
        sma_oi = df['total_oi'].rolling(30).mean()
        std_oi = df['total_oi'].rolling(30).std()
        avg_call_change = df['call_oi_chng'].rolling(30).mean()
        avg_put_change = df['put_oi_chng'].rolling(30).mean()
        pcr_vol = df['put_vol'] / df['call_vol']
        avg_pcr_vol = pcr_vol.rolling(30).mean()
        pcr_oi = df['put_oi'] / df['call_oi']
        avg_pcr_oi = pcr_oi.rolling(30).mean()
        
        
        #### Insert Moving Averages 
        # df.insert(1, 'group', df.stock.map(sg)) 
        df.insert(3, 'avg_oi', sma_oi)
        df.insert(6, 'total_oi_std', std_oi)
        df.insert(6, 'avg_call_change', avg_call_change)
        df.insert(9, 'avg_put_change', avg_put_change)
        df.insert(3, 'avg_vol', sma_vol)    
        df.insert(4, 'pcr_vol', pcr_vol)
        df.insert(5, 'avg_pcr_vol', avg_pcr_vol)
        df.insert(5, 'pcr_oi', pcr_oi)
        df.insert(6, 'avg_pcr_oi', avg_pcr_oi)
        
        # Drop NAN and Convert to Integers
        df.dropna(inplace = True)
        df['avg_vol'] = df['avg_vol'].astype(int)
        df['total_vol'] = df['total_vol'].astype(int)
        df['avg_oi'] = df['avg_oi'].astype(int)
        df['total_oi'] = df['total_oi'].astype(int)
        df['avg_call_change'] = df['avg_call_change'].astype(int)
        df['avg_put_change'] = df['avg_put_change'].astype(int)
        
    
    def _all_cp(self):
        """
        Returns the Daily Option Stats for all stocks 
        
        
        """
        sg = self.reverse_dict()    
        pbar = tqdm(self.stocks['all_stocks'], desc = "CP...")
        out = []
        for stock in pbar:
            try:
                df = pd.read_sql(f'''select * from "{stock}" order by datetime(gatherdate) asc''', self.vol_db, parse_dates = ['gatherdate'])
                df.gatherdate = pd.to_datetime(df.gatherdate, format = '%Y-%m-%dT%H:%M:%S')
                df.insert(0, 'stock', stock)
                out.append(df.tail(1))
            except:
                pass
        push_df = pd.concat(out, axis = 0)
        push_df.to_sql('daily_option_stats', self.stats_db, if_exists = 'replace', index = False)   
        return push_df
        
    
    
    
if __name__ == "__main__":
    print("Control what you can Control.")

    sys.path.append('/Users/jerald/Documents/Dir/Python/stocker')
    from bin.main import get_path
    connections = get_path()
    oc = Stats(connections)
    print(oc._all_cp())
