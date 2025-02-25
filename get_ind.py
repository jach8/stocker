""" 
Return the last row of the indicator dataframe for each stock
    - Also can return the full indicator df, for each stock in the form of a dictionary.
"""

import pandas as pd
from bin.price.db_connect import Prices
from bin.alerts.iterator import Iterator
from bin.main import get_path
from tqdm import tqdm 


class LastClose(Prices):
    def __init__(self, connections):
        super().__init__(connections)
        self.it = Iterator(connections)
    
    def daily_close(self, stock):
        """
        Get the last row of the daily close. 
        """
        out = self.get_indicators(stock, daily=True)
        out.insert(0, 'stock', stock)
        return out.tail(1)
    
    def intraday_close(self, stock):
        """
        Get the last row of the intraday close. 
        """
        out = self.get_indicators(stock, daily=False)
        out.insert(0, 'stock', stock)
        return out.tail(1)

    def last_close(self, daily = True):
        """
        Get the last close for each stock. 
        """
        if daily:
            close = self.daily_close
        else:
            close = self.intraday_close
            
        df = self.it.dataframe_iterator_function(close)
        return df

    def quick_average(self, match_string, ddf):
        """
        Get the average of the columns that contain the match_string.
        """
        cols = ddf.columns
        c = cols[ddf.columns.str.contains(match_string)]
        if match_string[-1] == '_':
            new_col = f'avg{match_string}'
        else:
            new_col = f'avg_{match_string}'
        ddf[new_col] = ddf[c].mean(axis=1)
        return ddf
    
    def difference_from_ma(self, df):
        """ Return the difference from the moving averages. """
        ma_columns = [x for x in df.columns if 'ma' in x]
        diff_df = df[ma_columns].copy()
        
        for col in ma_columns:
            diff_df[col] = df['close'] - df[col]
        
        diff_df.insert(0, 'stock', df['stock'])
        diff_df.insert(1, 'close', df['close'])
        diff_df['avg_diff'] = diff_df[ma_columns].mean(axis=1)
        avg_cols = ['_slow', '_fast', '_med', 'ema', 'sma', 'kama']
        for col in avg_cols:
            diff_df = self.quick_average(col, diff_df)
    

        return diff_df

    def full_indicators(self, stock):
        """ Return the full indicator dataframe for a stock. """
        out = self.Pricedb.get_indicators(stock)
        out.insert(0, 'stock', stock)
        return out

    def historical_last_close(self):
        """ 
        Get the full indicator dataframe for each stock in the form of a dictionary.
        """
        d = {}
        st = self.Optionsdb.stocks['all_stocks']
        for stock in tqdm(st):
            d[stock] = self.full_indicators(stock)
        return d

if __name__ == "__main__":
    from bin.main import get_path
    connections = get_path()
    lc = LastClose(connections)
    df = lc.last_close()
    print(df)
    print(lc.difference_from_ma(df))
    print(lc.historical_last_close)