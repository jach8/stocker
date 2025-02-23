"""
Earnings Data Gathering.


"""


import pandas as pd 
import numpy as np 
from pickle import load, dump
import datetime as dt 
import json 
import sys
sys.path.append('/Users/jerald/Documents/Dir/Python/Stocks')
from data.stocks.util.sector_util import get_Company_info
from tqdm import tqdm 
import sqlite3 as sql

class AddStocks:
    def __init__(self, connections):
        
        self.stock_info_path = connections['stock_info_dict']
        self.stock_names_db = sql.connect(connections['stock_names'])
        self.stocks = json.load(open(connections['ticker_path'], 'r'))['equities']

    def _update_equity_sector(self, d, write = True):
        """ 
        Update the Equity Sector Database. 
        """
        stock_info = d.copy()
        stocks  = list(d.keys())
        
            
        lodf = []; error = []
        for x in tqdm(stocks, desc = "Working: "): 
            try:
                a = pd.DataFrame.from_dict({x:y for x, y in d[x].items() if type(y) != list}, orient = 'index').T
                a.insert(0, 'stock', x)
                lodf.append(a)
            except Exception as e:
                error.append(x)
                pass
                

        lodf2 = [pd.DataFrame(x.iloc[0]).T for x in lodf ]
        df = pd.concat(lodf2).reset_index(drop=True)
        df.columns = [x.lower() for x in df.columns]

        df.beta = df.beta.astype(float) 
        df.forwardpe = df.forwardpe.astype(float)
        df.marketcap = df.marketcap.astype(float)
        c = ['industry', 'sector', 'industrykey', 'sectorkey']
        
        # Adds the new table nameed sector into the database. 
        ind_sect = df.set_index('symbol').copy()
        ind_sect = ind_sect.sort_values('sector', ascending=True)[c].reset_index().copy()
        if write == True: 
            ind_sect.dropna().to_sql('sectors', self.stock_names_db, if_exists='replace', index=False)
        cols = list(df.columns)
        if write == True: df[df.phone.notnull()][cols].to_sql('equity_info', self.stock_names_db, if_exists='replace', index=False)

        # ETF Info
        df_etf = df[df.phone.isnull()].copy()
        df_etf.columns = [x.lower() for x in df_etf.columns]
        if write == True: df_etf.to_sql('etf_info', self.stock_names_db, if_exists='replace', index=False)
        
        return df, df_etf
    
    def _lowest_prict_to_book(self, df):
        c = [
        'pricetobook', 'marketcap', 'sharesoutstanding','enterprisevalue', 'floatshares', 
        'dividendyield', 'forwardpe', 'marketcap', 'profitmargins', 
        'grossmargins', 'operatingmargins', 'ebitda', 
        'totalrevenue', 'debttoequity', 'returnonassets' ,'returnonequity',
        'freecashflow', 'operatingcashflow', 'trailingpegratio', 'overallrisk',
        ]
        cry = (df.marketcap > 4e9)

        best = df[cry].set_index('symbol').sort_values('pricetobook', ascending=True).head(20)[c]
        print(best)
        
    
    def _update_company_info(self):
        """ 
        Update Company info JSON file. 
        """
        d = get_Company_info(self.stocks, self.stock_info_path)
        equity, etf = self._update_equity_sector(d)
        self._lowest_prict_to_book(equity)
        return d 
    
    def _update_info_no_download(self):
        """ 
        Update the company info without downloading the data. 
        """
        d = json.load(open(self.stock_info_path, 'r'))
        equity, etf = self._update_equity_sector(d)
        print(equity['marketcap'])
    
    def load_stock_info(self):
        """ 
        Load the stock info. 
        """
        d = json.load(open(self.stock_info_path, 'r'))
        return d
        
        
if __name__ == "__main__":
    import sys
    sys.path.append('/Users/jerald/Documents/Dir/Python/Stocks')
    from bin.main import get_path 
    
    connections = get_path()
    add = AddStocks(connections)
    add._update_info_no_download()
    # add._update_company_info()