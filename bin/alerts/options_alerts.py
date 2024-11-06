import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import datetime as dt 
from tqdm import tqdm 
import scipy.stats as st 
import time
import json 

import sys
sys.path.append('/Users/jerald/Documents/Dir/Python/stocker')



class Notifications:
    def __init__(self, connections):
        self.vol_db = connections['vol_db']
        self.stocks = json.load(open(connections['ticker_path'], 'r'))
        

    def stock_data(self, stock, n = 5, date = None):
        """ 
        Return the daily Option Statistics for a given stock. 
            Also calculate the 5 day net change in open interest and volume for calls and puts. 
            
        Args:
            stock: str
            n: int (optional) --> Lookback period for the net change in open interest and volume
            date: str (optional) --> Get data for a specific date
        """
        try: 
            if date == None:
                df = pd.read_sql(f'select * from "{stock}"', self.vol_db, parse_dates = ['gatherdate'], index_col=['gatherdate'])
            else:
                df = pd.read_sql(f'select * from "{stock}" where date(gatherdate) <= date("{date}")', self.vol_db, parse_dates = ['gatherdate'], index_col=['gatherdate'])
            
            # drop columns that have pct in them 
            dropCols = list(df.filter(regex='pct|spread|delta|gamma|theta|vega|prem|iv|total'))
            df = df.drop(columns=dropCols)
            
            # keepCols = list(df.filter(regex='iv|vol|oi|chng'))
            
            # df['call_oi_chng5d'] = df['call_oi'].diff(5)
            # df['put_oi_chng5d'] = df['put_oi'].diff(5)
            # df['call_vol_chng5d'] = df['call_vol'].diff(5)
            # df['put_vol_chng5d'] = df['put_vol'].diff(5)        
            
            
            return df 
        except:
            return pd.DataFrame()
    
    def _percentile_score(self, df, col):
        """ 
        returns a percentile score for the column.
        
        To be interpreted as the percentage of values in the column that are less than the last observation. 
            - A score of 0 means that 100% of the values are less than the last observation. (Anomalous Behavior, unusually high)
            - A score of 50 means that 50% of the values are less than the last observation. (Normal Behavior)
            - A score of 99 means that 1% of the values are less than the last observation. (Anomalous Behavior, unusually low)
        
        Args:
            df: DataFrame
            col: column name
        """
        return st.percentileofscore(df[col], df[col].iloc[-1], kind = 'strict', nan_policy='omit')
    
    def col_map(self, col):
        cmap = {
            "call_vol": "Call Volume",
            "put_vol": "Put Volume",
            "total_vol": "Total Volume",
            "call_oi": "Call OI",
            "put_oi": "Put OI",
            "total_oi": "Total OI",
            "call_prem": "Call Premium",
            "put_prem": "Put Premium",
            "total_prem": "Total Premium",
            "call_iv": "Call IV",
            "put_iv": "Put IV",
            "atm_iv": "ATM IV",
            "otm_iv": "OTM IV",
            "call_vol_chng": "Call Volume Chng",
            "put_vol_chng": "Put Volume Chng",
            "total_vol_chng": "Total Volume Chng",
            "call_oi_chng": "Call OI Chng",
            "put_oi_chng": "Put OI Chng",
            "total_oi_chng": "Total OI Chng",
            "call_prem_chng": "Call Prem. Chng",
            "put_prem_chng": "Put Prem. Chng",
            "total_prem_chng": "Total Prem. Chng",
            "call_iv_chng": "Call IV Chng",
            "put_iv_chng": "Put IV Chng",
            "atm_iv_chng": "ATM IV Chng",
            "otm_iv_chng": "OTM IV Chng",
            "call_oi_chng5d": "Call OI Chng (5d)",
            "put_oi_chng5d": "Put OI Chng (5d)",
            "call_vol_chng5d": "Call Vol Chng (5d)",
            "put_vol_chng5d": "Put Vol Chng (5d)",
        }
        
        return cmap[col]
    
    def _colors(self, color = None, word = None):
        header_text = '\033[95m'
        red_text = '\033[91m'
        green_text = '\033[92m'
        yellow_text = '\033[93m'
        blue_text = '\033[94m'
        orange_text = '\033[96m'
        pink_text = '\033[93m'
        end_text = '\033[0m'
        
        names = ['header', 'red', 'green', 'yellow', 'blue','orange','pink','end']
        colors = [header_text, red_text, green_text, yellow_text, blue_text, orange_text, pink_text, end_text]
        c = dict(zip(names, colors))
        if color == None:
            return c
        else:
            if word == None:
                return c[color]
            else:
                return c[color] + word + end_text
            
        
    def __generate_text(self, stock, df, col):
        """ 
        Return the text notifications for a given stock.
        """
        header_text = '\033[95m'
        red_text = '\033[91m'
        green_text = '\033[92m'
        yellow_text = '\033[93m'
        blue_text = '\033[94m'
        orange_text = '\033[33m'   
        pink_text = '\033[95m'
        end_text = '\033[0m'
        
        
        X = np.abs(df[col].iloc[-1])
        X_mu = df[col].iloc[:-1].mean()
        col_max = df[col].iloc[:-1].max()
        col_min = df[col].iloc[:-1].min()
        perc = self._percentile_score(df.abs(), col) / 100
        lower_25 = df[col].quantile(.05)
    
        def callPut_text(x):
            if 'Call' in x:
                return f'🟢 {green_text}{x}{end_text}'
            if 'Put' in x:
                return f'🔴 {red_text}{x}{end_text}'
            if 'IV' in x:
                return f'🟡 {blue_text}{x}{end_text}'
            if 'Total' in x:
                return f'⚫️ {x}'
            else:
                return x
            
        txt = None
        cName = self.col_map(col)
        
        ### All time high behavior ###
        if np.abs(X) > col_max:
            txt = callPut_text(f'${stock.upper()} ' + cName + ' ')
            if 'prem' in col:
                txt += self._colors('green', f'${X:,.2f} ') + f'(𝜇 ＝ ${X_mu:,.2f}) '
            
            if 'vol' in col:
                txt += self._colors('blue', f'{X:,.0f} ') + f'(𝜇 ＝ {X_mu:,.0f}) '
            
            if 'oi' in col:
                txt += self._colors('blue', f'{X:,.0f} ') + f'(𝜇 ＝ {X_mu:,.0f}) '
            
            if 'iv' in col:
                txt += self._colors('yellow', f'{X:.2%} ') + f'(𝜇 ＝ {X_mu:.2%}) '
            # txt += f'highest in {df.shape[0]} days ' + self._colors('orange', f'{100 * perc:.2f}th%')
            # txt += f'highest in' + self._colors('orange', f' {df.shape[0]} days ')
        
        ###### Unusually Low behavior ###### 
        if X <= lower_25 and X > 0 and perc < 0.03:
        
            if 'prem' in col:
                txt = f'   ${stock.upper()} ' + self._colors( 'red',cName + ' ')
                txt += self._colors('red', f'${X:,.2f} (𝜇 ＝ ${X_mu:,.2f}) ') + f' (bottom \33[1m {perc:.2%})' + self._colors('end')
            
            if 'iv' in col:
                txt = f'   ${stock.upper()} ' + self._colors( 'red',cName + ' ')
                txt += self._colors('red', f'{X:.2%} (𝜇 ＝ {X_mu:.2%}) ') + f' (bottom \33[1m {perc:.2%})' + self._colors('end')
            

                
        if txt != None:
            return txt
        else:
            None
            
            
    def __remove_colors(self, txt):
        """ 
        Remove the colors from the text. 
        """
        c = [x[1] for x in list(self._colors().items())]
        
        for i in c:
            if i in txt:
                txt = txt.replace(i, '')
        return txt
        
        
    def notifications(self, stock, n = 5, date = None):
        """ 
        Return the text notifications for a given stock. 
        """
        
        c_out = []
        out = []
        if stock not in self.stocks['all_stocks']:
            return out 
        df = self.stock_data(stock, n, date)
        cols = list(df.columns)
        if df.shape[0] > 100:
            for col in cols:
                txt = self.__generate_text(stock, df, col)
                if txt != None:
                    c_out.append(txt)

            if c_out != []: 
                for x in c_out: 
                    print(x)
                    out.append(self.__remove_colors(x))
            return out
        else:
            return out
    
    
    def iterator(self, n = 5, date = None):
        """ 
        Iterate through the stocks and return the notifications. 
        """
        
        out = []
        stocks = self.stocks['all_stocks']
        for i in tqdm(stocks): 
            j = self.notifications(i, n, date)
            if  j != []:
                for k in j:
                    out.append(k)  
                    time.sleep(.01)
        return out
    
if __name__ == '__main__':
    from bin.main import get_path
    connections = get_path()
    notif = Notifications(connections)
    # dates = pd.bdate_range('2024-07-29', '2024-08-05')
    # for i in dates:
    #     try:
    #         print(i)
    #         ot = notif.iterator(date = i)
    #         print('\n\n')
    #     except:
    #         pass
    out = notif.iterator(n = 3, date = "2024-10-17")