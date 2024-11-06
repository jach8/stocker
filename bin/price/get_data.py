import pandas as pd 
import numpy as np
import sqlite3 as sql
import yfinance as yf
import pandas_datareader.data as web
import datetime
import json

class update_stocks: 
    def __init__(self, connections):
        self.stocks_db = connections['daily_db']
        self.stocks_intraday_db = connections['intraday_db']
        self.ticker_path = connections['ticker_path']
        
    def stock_names(self):
        stocks = json.load(open(self.ticker_path, 'r'))
        return stocks['all_stocks']
        
    def update_stocks(self):
        c = sql.connect(self.stocks_db)
        stocks = self.stock_names()
        s = ' '.join(stocks)
        # d = pd.read_sql_query('select date(max(Date)) from spy', c).iloc[0][0]
        d = '1997-02-07'
        data = yf.download(s, start = d)
        data = data.swaplevel(0,1, axis = 1).sort_index(axis = 1)
        stocksU = [s.upper() for s in stocks]
        d = {s:data[s].drop_duplicates() for s in stocksU}
        for s, u in enumerate(stocks):
            db_add = d[stocksU[s]]
            db_add = db_add[~db_add.index.duplicated(keep='last')].dropna()
            db_add.to_sql(stocksU[s], con=c, if_exists='replace')
        print('Stocks Updated (Daily Data)')
    
    def update_stocks_intraday(self):
        c = sql.connect(self.stocks_db)
        stocks = self.stock_names()
        s = ' '.join(stocks)
        conn = sql.connect(self.stocks_intraday_db)
        data = yf.download(' '.join(stocks), period = '5d', interval = '1m')
        data = data.swaplevel(0,1, axis = 1).sort_index(axis = 1)
        data.index = [str(x).split('-04:00')[0] for x in data.index]
        #data.index = pd.to_datetime(data.index, format = '%Y-%m-%d %H:%M:%S')
        stocksU = [s.upper() for s in stocks]
        d = {s:data[s].drop_duplicates() for s in stocksU}
        for s, u in enumerate(stocks):
            db_add = d[stocksU[s]].copy()
            db_add = db_add[~db_add.index.duplicated(keep='last')].dropna()
            db_add = db_add.reset_index()
            db_add.rename(columns = {'index':'Date'}, inplace = True)
            db_add['Date'] = [str(x)[:19] for x in db_add.Date]
            db_add['Date'] = db_add['Date'].str.replace('T', ' ')
            db_add['Date'] = pd.to_datetime(db_add.Date)
            db_add.to_sql(stocksU[s], con=conn, if_exists='append', index = False)
        print('Stocks Updated (Intraday Data)')

    def update(self):
        self.update_stocks()
        self.update_stocks_intraday()


if __name__ == '__main__':
    print("Updating Stock Price Database..")    
    
    connections = {
        'stocks_db': 'data/prices/stocks.db',
        'stocks_intraday_db': 'data/prices/stocks_intraday.db',
        'ticker_path': 'data/stocks/tickers.json'
    }
    
    price_update = update_stocks(connections)
    price_update.update() 