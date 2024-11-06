################################################################################################
"""
            Scanning Module for Finding Profitable Contracts
            ---------------------------------------------------
    1. This module is designed to scan through a list of stocks and find profitable contracts.
    2. We want to return a dictionary with flexible keys such as: 
        1. Top Volume Gainers 
        2. Top Open Interest Gainers
        3. Highest IV compared to 30 day averageq
        4. Highest Percentage Gain in Price. 
        5. Highest Volume to Open interest ratio 
        ....
        -- Etc. 
        We want to keep this flexibile so that later we can add more screening methods, like outputs from a machine learning model 

"""
################################################################################################
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import datetime as dt 
import json 


import sys
sys.path.append('/Users/jerald/Documents/Dir/Python/stocker')

from bin.signals.iterator import Iterator


class Scanner(Iterator):
    def __init__(self, connections = None):
        super().__init__(connections)
        self.stock_dict = json.load(open(connections['ticker_path'], 'r'))
        self.todays_date = dt.datetime.today().date()
    
        
    def pct_chg(self, stock):
        """ Highest Percentage Gains in Contract price. 
                Looks for 1,000% gains in an option contract. 
        """
        q = f''' 
        select 
            --contractsymbol, 
            --datetime(gatherdate) as gatherdate
            *
        from "{stock}" 
        where datetime(gatherdate) = (select max(datetime(gatherdate)) from "{stock}")
        and pct_chg > 1000
        and ask > 0.1
        and bid > 0.1
        and (ask - bid) < 0.10
        order by pct_chg desc
        limit 5
        '''
        return q
    
    def volume(self, stock):
        """ Highest Volume Contracts """
        q = f''' 
        select 
            --contractsymbol, 
            --datetime(gatherdate) as gatherdate
            *
        from "{stock}" 
        where datetime(gatherdate) = (select max(datetime(gatherdate)) from "{stock}")
        and volume > 1000
        and ask > 0.1
        and bid > 0.1
        and (ask - bid) < 0.10
        and ask < 2
        and bid < 2
        order by volume desc
        limit 5
        '''
        return q
    
    
    def voi(self, stock):
        """ Highest Volume to Open Interest Ratio """
        q = f''' 
        select 
            --contractsymbol, 
            --datetime(gatherdate) as gatherdate
            *,
            volume / openinterest as voi
        from "{stock}" 
        where datetime(gatherdate) = (select max(datetime(gatherdate)) from "{stock}")
        and volume > 1000
        and volume / openinterest > 1
        and ask > 0.1
        and bid > 0.1
        and (ask - bid) < 0.10
        order by volume / openinterest desc
        limit 5
        '''
        return q
    
    def iv_diff(self, stock):
        """ Highest IV Ranking, comparing current IV to the 30 day average. """
        q = f''' 
        select 
            --contractsymbol, 
            --datetime(gatherdate) as gatherdate
            *, 
            iv_avg_30d - impliedvolatility as iv_diff
        from "{stock}" 
        where datetime(gatherdate) = (select max(datetime(gatherdate)) from "{stock}")
        and impliedvolatility < iv_avg_30d
        and ask > 0.1
        and bid > 0.1
        and (ask - bid) < 0.10
        and ask < 2
        and bid < 2
        order by (iv_avg_30d - impliedvolatility) desc
        limit 5
        '''
        return q
    
    def oi_chg(self, stock):
        """ Highest Open Interest Change """
        q = f''' 
        select 
            --contractsymbol, 
            --datetime(gatherdate) as gatherdate
            *
        from "{stock}" 
        where datetime(gatherdate) = (select max(datetime(gatherdate)) from "{stock}")
        and oi_chg > 100
        and ask > 0.1
        and bid > 0.1
        and (ask - bid) < 0.10
        and ask < 2
        and bid < 2
        order by oi_chg desc    
        limit 5
        '''
        return q
    
    def amnt(self, stock):
        """ 
        Highest 'AMNT' Change, this is when the change of open interest is greater than the total volume from the previous day.
                - This implies that there were contracts traded after the market closed, OR that the real volume was not reported. 
        """
        q = f'''
        select 
            --contractsymbol, 
            --datetime(gatherdate) as gatherdate
            *
        from "{stock}"
        where datetime(gatherdate) = (select max(datetime(gatherdate)) from "{stock}")
        and amnt > 0 
        and ask > 0.1
        and bid > 0.1
        and (ask - bid) < 0.10
        and ask < 2
        and bid < 2
        order by amnt desc
        limit 5
        '''
        return q
    
    
    def run(self):
        # return self.group_query_iterator(self.high_percent_changes, self.connection, group = 'all_stocks')
        # return self.query_iteroator(self.high_percent_changes, self.connection, group = 'all_stocks')
        list_of_functions = [self.pct_chg, self.volume, self.voi, self.iv_diff, self.oi_chg, self.amnt]

        return self.list_iterator(list_of_functions, conn = 'change_db', group = 'all_stocks')

    def scan_contracts(self):
        out = self.run()
        table_names = list(out.keys())
        write_connection = self.get_connection('stats_db')
        for table in table_names:
            df = out[table].copy()
            df.to_sql(table, write_connection, if_exists = 'replace', index = False)



if __name__ == "__main__":
    from bin.main import get_path
    connections = get_path()
    sc = Scanner(connections)
    print(sc.test_run())
    
    
    
        
    