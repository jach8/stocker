import sys
sys.path.append('/Users/jerald/Documents/Dir/Python/stocker')



import sqlite3 as sql
import pandas as pd 
import numpy as np 


class daily_option_stats:
    """
    Daily Option Statistics For all stocks 
    
    """
    def __init__(self, connections:dict):
        self.vol_db = sql.connect(connections['vol_db'])
        print('\n\n')
        self.cmap = {
                "call_vol": ("Call Volume", "number"),
                "put_vol": ("Put Volume", "number"),
                "total_vol": ("Total Volume", "number"),
                "call_oi": ("Call Open Interest", "number"),
                "put_oi": ("Put Open Interest", "number"),
                "total_oi": ("Total Open Interest", "number"),
                "call_prem": ("Call Premium", "dollars"),
                "put_prem": ("Put Premium", "dollars"),
                "total_prem": ("Total Premium", "dollars"),
                "call_iv": ("Call Implied Volatility", "percentage"),
                "put_iv": ("Put Implied Volatility", "percentage"),
                "atm_iv": ("At-The-Money Implied Volatility", "percentage"),
                "otm_iv": ("Out-of-The-Money Implied Volatility", "percentage"),
                "call_vol_chng": ("Call Volume Change", "number"),
                "put_vol_chng": ("Put Volume Change", "number"),
                "total_vol_chng": ("Total Volume Change", "number"),
                "call_oi_chng": ("Call Open Interest Change", "number"),
                "put_oi_chng": ("Put Open Interest Change", "number"),
                "total_oi_chng": ("Total Open Interest Change", "number"),
                "call_prem_chng": ("Call Premium Change", "dollars"),
                "put_prem_chng": ("Put Premium Change", "dollars"),
                "total_prem_chng": ("Total Premium Change", "dollars"),
                "call_iv_chng": ("Call Implied Volatility Change", "percentage"),
                "put_iv_chng": ("Put Implied Volatility Change", "percentage"),
                "atm_iv_chng": ("ATM Implied Volatility Change", "percentage"),
                "otm_iv_chng": ("OTM Implied Volatility Change", "percentage"),
                "call_oi_chng5d": ("Call Open Interest Change (5 days)", "number"),
                "put_oi_chng5d": ("Put Open Interest Change (5 days)", "number"),
                "call_vol_chng5d": ("Call Volume Change (5 days)", "number"),
                "put_vol_chng5d": ("Put Volume Change (5 days)", "number"),
            }
            
    def col_map(self, col):
        """
        
        
        """
        return self.cmap[col][0]
    
    def number_format(self, col): 
        """
        
        
        """
        map = {'number': '{:,.0f}', 'percentage': '{:.2%}', 'dollars': '${:,.2f}'}
        return map[self.cmap[col][1]]

    def stock_data(self, stock, date = None):
        """ 
        Return the daily Option Statistics for a given stock. 
            
        Args:
            stock: str
            date: str (optional) --> Get data for a specific date
        """
        if date == None:
            df = pd.read_sql(f'select * from {stock}', self.vol_db, parse_dates = ['gatherdate'], index_col=['gatherdate'])
        else:
            df = pd.read_sql(f'select * from {stock} where date(gatherdate) <= date("{date}")', self.vol_db, parse_dates = ['gatherdate'], index_col=['gatherdate'])
        
        # drop columns that have pct in them 
        df = df.sort_index()
        dropCols = list(df.filter(regex='pct|spread|delta|gamma|theta|vega|prem|iv|total|chng'))
        df = df.drop(columns=dropCols)
        return df 
    
    def all_time_highs(self, stock, df, col):
        """ 
        if the current data is at an all time high, return true. 
        """
        current_data = df[col].iloc[-1]
        max_val = df[col].abs().max()
        cname = self.col_map(col)
        cformat = self.number_format(col)
        # Need to take the absolute value of the current data to compare with the absolute value of the max value
        if np.abs(current_data) >= max_val:
            print(f'${stock.upper()}:\t{cname} is at an all time high of {cformat.format(max_val)}')
            return f'{cname} is at an all time high of {cformat.format(max_val)}'
        return None
    
    def all_time_lows(self, stock, df, col):
        """ 
        if the current data is at an all time low, return true. 
        """
        current_data = df[col].iloc[-1]
        min_val = df[col].abs().min()
        cname = self.col_map(col)
        cformat = self.number_format(col)
        # Need to take the absolute value of the current data to compare with the absolute value of the max value
        if np.abs(current_data) <= min_val:
            print(f'${stock.upper()}:\t{cname} is at an all time low of {cformat.format(min_val)}')
            return f'{cname} is at an all time low of {cformat.format(min_val)}'
        return None
    
    def highest_in_x_days(self, stock, df, col, days = 30):
        """ 
        if the current data is at an all time high, return true. 
        """
        df = df.sort_index().tail(days)
        days = len(df)
        current_data = df[col].iloc[-1]
        max_val = np.abs(df[col].max())
        cname = self.col_map(col)
        cformat = self.number_format(col)
        if current_data >= max_val:
            print(f'${stock.upper()}:\t{cname} of {cformat.format(max_val)} at highest in {days} days')
            return f'{cname} of {cformat.format(max_val)} at highest in {days} days'
        return None
    
    def lowest_in_x_days(self, stock, df, col, days = 30):
        """ 
        if the current data is at an all time high, return true. 
        """
        df = df.sort_index().tail(days)
        days = len(df)
        current_data = df[col].iloc[-1]
        min_val = np.abs(df[col].min())
        cname = self.col_map(col)
        cformat = self.number_format(col)
        if current_data <= min_val:
            print(f'${stock.upper()}:\t{cname} of {cformat.format(min_val)} at lowest in {days} days')
            return f'{cname} of {cformat.format(min_val)} at lowest in {days} days'
        return None
    
    def _column_iterator(self, stock, date = None):
        df = self.stock_data(stock, date)
        alerts = []
        for col in df.columns:
            # alerts.append(self.all_time_highs(stock, df, col))
            # alerts.append(self.all_time_lows(stock, df, col))
            alerts.append(self.highest_in_x_days(stock, df, col, days = 100))
            alerts.append(self.lowest_in_x_days(stock, df, col, days = 100))
        return alerts
        
    def _iterator(self, date = None):
        """
        
        
        """
        stocks = self.stock_dict['all_stocks']
        alerts = []
        for stock in stocks:
            alerts.append(self._column_iterator(stock, date))
            print('\n')
        return alerts
    
if __name__ == '__main__':
    from bin.main import get_path   
    connections = get_path()
    notif = daily_option_stats(connections)

    out = notif._iterator()