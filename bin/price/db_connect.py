import numpy as np 
import pandas as pd 
import sqlite3 as sql 
import datetime as dt 
from tqdm import tqdm
import time
import json 

import sys
sys.path.append('/Users/jerald/Documents/Dir/Python/stocker')
from bin.price.indicators import Indicators
from bin.price.get_data import update_stocks

class Prices(update_stocks):
    def __init__(self, connections):
        super().__init__(connections)
        self.execution_start_time = time.time()
    
        try:
            self.names_db = sql.connect(connections['stock_names'])
            self.daily_db = sql.connect(connections['daily_db'])
            self.intraday_db = sql.connect(connections['intraday_db'])
            self.stocks =  json.load(open(connections['ticker_path'], 'r'))['all_stocks']
            ## Dict of cursurs for [daily_db, intraday_db]
            self.Indicators = Indicators 
            
            print("Prices Connected: {}".format(dt.datetime.now()))
        except Exception as e:
            print("Connection Failed: ", e,)
            
    def update_stock_prices(self):
        self.update()
            
    def custom_q(self, q):
        """ Execute a qustom query on the daily_db """
        cursor = self.daily_db.cursor()
        cursor.execute(q)
        return pd.DataFrame(cursor.fetchall(), columns = [desc[0] for desc in cursor.description])
            
    def _get_stock_sectors(self):
        names_df = pd.read_sql('SELECT lower(symbol) as stock, sectorkey, industrykey FROM sectors', self.names_db)
        return names_df 
     
    def _get1minCl(self, stock):
        q = f'''select datetime(date) as date, close from "{stock}" order by datetime(date) asc'''
        cursor = self.intraday_db.cursor()
        cursor.execute(q)
        df = pd.DataFrame(cursor.fetchall(), columns = ['date', stock])
        df.date = pd.to_datetime(df.date)
        return df.set_index('date')
    
    def get_intraday_close(self, stocks, agg = '1min'):
        """ Input a List of Stocks to obtain the closing prices for each stock """
        assert isinstance(stocks, list), "Input must be a list of stocks"
        out = [self._get1minCl(stock) for stock in stocks]
        out = [i.resample(agg).last() for i in out]
        return pd.concat(out, axis = 1)
        
    def _getClose(self, stock):
        q = f'''select date(date) as date, "adj close" as "Close" from "{stock}" order by date(date) asc'''
        df = pd.read_sql_query(q, self.daily_db, parse_dates = ['date'], index_col='date')
        return df.rename(columns = {'Close':stock})
    
    def get_close(self, stocks, start = None, end = None):
        """ Input a List of Stocks to obtain the closing prices for each stock """
        assert isinstance(stocks, list), "Input must be a list of stocks"
        out = [self._getClose(stock) for stock in stocks]
        return pd.concat(out, axis = 1,)
    
    def _normalize(self, df):
        """ Return the normalized price of a dataframe. 
                df: DataFrame with date as index. 
        """
        return df / df.iloc[0]
    
    def _returns(self, df):
        """ Return the returns of a dataframe """
        return df.ffill().pct_change()
    
    def ohlc(self, stock, daily = True, start = None, end = None):
        if daily == True:
            q = f'''select date(date) as "Date", open, high, low, close, volume from "{stock}" order by date(date) asc'''
            df = pd.read_sql_query(q, self.daily_db, parse_dates = ['Date'], index_col='Date')
        else: 
            q = f'''select datetime(date) as "Date", open, high, low, close, volume from "{stock}" order by datetime(date) asc'''
            df = pd.read_sql_query(q, self.intraday_db, parse_dates = ['Date'], index_col='Date')
        if start is not None:
            df = df[df.index >= start]
        if end is not None:
            df = df[df.index <= end]
        return df
    
    def _exclude_duplicate_ticks(self, group):
        """ Exclude duplicate tickers """
        if group == 'etf':
            g = list(set(self.stocks['etf']) - set(self.stocks['market']) - set(self.stocks['bonds']))
        else:
            g = self.stocks[group]
        return g
    
    def sectors(self):
        """ Return Sector Performance for the Equities we track """
        names_df = self._get_stock_sectors()
        sector_performance = pd.DataFrame()
        sect = list(names_df.sectorkey.unique())

        for s in sect:
            sk = names_df[names_df.sectorkey == s].stock.values
            prices = self.get_close(list(sk)).ffill().bfill()
            returns = prices.pct_change().dropna() 
            ## Sector Performance 
            sector_performance[s] = prices.mean(axis = 1)
        return sector_performance
    
    def industries(self):
        """ Return Industry Performance for the Equities we track """
        names_df = self._get_stock_sectors()
        industry_performance = pd.DataFrame()
        sect = list(names_df.industrykey.unique())

        for s in sect:
            sk = names_df[names_df.industrykey == s].stock.values
            # print(list(sk))
            prices = self.get_close(list(sk)).ffill().bfill()
            returns = prices.pct_change().dropna() 
            ## Sector Performance 
            industry_performance[s] = prices.mean(axis = 1)
        return industry_performance
    
    def get_aggregates(self, df):
        """ 
            Return Daily, Weeky, Monthly Aggregates for data with a date index 
                args:
                    df: DataFrame with a datetime index
                returns:
                    dict: {'daily':df, 'weekly':df, 'monthly':df}
        """
        assert isinstance(df.index, pd.DatetimeIndex), "Index must be a DatetimeIndex"
        daily = df.resample('B').last().dropna()
        weekly = df.resample('W').last().dropna()
        monthly = df.resample('M').last().dropna()
        return {'B':daily, 'W':weekly, 'M':monthly}
    
    def intra_day_aggs(self, df):
        """
            Return 3 min, 6 min, 18 min, 28 min, 1 hour, 4 hour aggregates for intraday data
        """
        df.index = pd.to_datetime(df.index)
        assert isinstance(df.index, pd.DatetimeIndex), "Index must be a DatetimeIndex"
        threes = df.resample('3T').last().dropna()
        sixes = df.resample('6T').last().dropna()
        eights = df.resample('18T').last().dropna()
        hours = df.resample('H').last().dropna()
        fours = df.resample('4H').last().dropna()
        return {'3min':threes, '6min':sixes, '18min':eights, '1H':hours, '4H':fours}
    
    def daily_aggregates(self, stock):
        df = self._getClose(stock)
        return self.get_aggregates(df)
    
    def get_indicators(self, stock, daily = True, kwargs = None, start = None, end = None):
        """ Return the indicators for a stock """
        if kwargs is None:
            kwargs = dict(fast = 6, medium = 10, slow = 28, m = 2)
        if daily == False:
            ############################## Get the Daily Moving Averages ####################
            daily_df = self.ohlc(stock, True, start, end)    
            G = Indicators(daily_df)
            daily_smas = G._get_moving_averages(fast = kwargs['fast'], medium = kwargs['medium'], slow = kwargs['slow'])
            dsma = pd.DataFrame(daily_smas, index = daily_df.index, columns = list(daily_smas.keys()))
            dsma.index = pd.to_datetime(dsma.index)
            colmaps = dict(_fast = str(kwargs['fast'])+"D",
                           _med = str(kwargs['medium'])+"D",
                           _slow = str(kwargs['slow'])+"D"
                    )
            fast_cols = dsma.columns.str.contains('_fast')
            medium_cols = dsma.columns.str.contains('_med')
            slow_cols = dsma.columns.str.contains('_slow')
            
            fc = {x:x.replace('_fast', colmaps['_fast']) for x in dsma.columns[fast_cols]}
            mc = {x:x.replace('_med', colmaps['_med']) for x in dsma.columns[medium_cols]}
            sc = {x:x.replace('_slow', colmaps['_slow']) for x in dsma.columns[slow_cols]}
            dsma.rename(columns = {**fc, **mc, **sc}, inplace = True)
            self.daily_smas = dsma
            
            ############################## Get the Indicators (Inter-day) ####################
            
            df = self.ohlc(stock, daily, start, end)
            i = Indicators(df)
            self.Indicators = i
            out = i.indicator_df(fast = kwargs['fast'], medium = kwargs['medium'], slow = kwargs['slow'], m = kwargs['m'])
            dsma['date'] = dsma.index.date
            out['date'] = out.index.date
            out['Date'] = out.index
            out = pd.merge(out, dsma, on = 'date', how = 'left').drop(columns = ['date'])
            return out.set_index('Date')
        else:    
            df = self.ohlc(stock, daily, start, end)
            i = Indicators(df)
            self.Indicators = i
            return i.indicator_df(fast = kwargs['fast'], medium = kwargs['medium'], slow = kwargs['slow'], m = kwargs['m'])
            
        
    def close_connections(self):
        for i in [self.names_db, self.daily_db, self.intraday_db]:
            i.close()
        end_time = time.time()
        runtime_min = (end_time - self.execution_start_time) / 60
        print("Connections Closed {}\nTotal Runtime: {:.2f} min".format(dt.datetime.now(), runtime_min))
        print()
        return None
    
    
if __name__ == "__main__":
    print("\n(26) To whatever and wherever the restless and unsteady mind wanders this mind should be restrained then and there and brought under the control of the self alone. (And nothing else) \n")
    connections = {
                ##### Prices ###########################
                'daily_db': 'data/prices/stocks.db', 
                'intraday_db': 'data/prices/stocks_intraday.db',
                'ticker_path': 'data/stocks/tickers.json',
                'stock_names' : 'data/stocks/stock_names.db'
    }
    
    m = Prices(connections)
    # print(m.all_stocks())
    print('\n\n\n')
    # print(m.get_indicators('aapl', daily = False, start = "2024-08-14"))    
    print(m.get_intraday_close(['aapl']))
    m.close_connections()
    
    