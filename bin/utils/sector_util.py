"""
Get updates for company information and sector keys. 
This should be used when you are adding stocks to the database. 

"""

from tqdm import tqdm 
import yfinance as yf 
import numpy as np 
import pandas as pd 
import json 
import time
import datetime as dt
import pickle




def get_Company_info(stocks, path):
    """ 
    Get the Company Information for the Stock 
    """
    d = {}
    for i in tqdm(stocks, desc = "Loading In Comapny Info"):
        try: 
            data = yf.Ticker(i).info
            data['date'] = dt.datetime.now().strftime('%Y-%m-%d')
            d[i] = data
        except:
            pass
        time.sleep(1)
        
    with open(path, 'w') as f:
        json.dump(d, f)
        
    print('done')
    return d
