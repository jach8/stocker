"""
Calculates the change in the option contracts. 
    - Read in the last 5 days from the vol.db
    - Calculate the new changes from the new option chain.
    - Update the changes in the vol.db

"""


import sys
sys.path.append('/Users/jerald/Documents/Dir/Python/stocker')
import pandas as pd 
import numpy as np 
import yfinance as yf 
import datetime as dt 

from bin.options.optgd.db_connect import Connector

class ChangeVars(Connector):
    def __init__(self, connections):
        super().__init__(connections)
        
    def _calc_changes(self, stock, N = None):
        if N == None:
            dte = 'date(gatherdate) > date("2022-11-17")'
        if N != None:
            recent_dates = self._last_dates(stock, N = N)
            dte = f'date(gatherdate) > date("{recent_dates[0]}")'
             
        q0 = f'''
            select 
            max(datetime(gatherdate)) as gatherdate,
            contractsymbol,  
            stk_price,
            lastprice,
            ask, 
            bid,
            change, 
            cast(percentchange as float) as percentchange,
            cast(ifnull(volume, 0) as int) as vol,
            cast(ifnull(openinterest, 0) as int) as oi,
            impliedvolatility
            from "{stock}"
            where {dte}
            --and ask > 0.10
            --and bid > 0.05
            --and abs(ask - bid) < 0.10
            group by contractsymbol, date(gatherdate)
            order by datetime(gatherdate) asc
            '''

        lags = f'over (partition by contractsymbol order by datetime(gatherdate))'
        # moving_avg = f'over(order by datetime(gatherdate) rows between 10 preceeding and current row)'
        moving_avg = f'over(partition by contractsymbol order by datetime(gatherdate) rows between 29 preceding and current row)'
        fast_moving_avg = f'over(partition by contractsymbol order by datetime(gatherdate) rows between 5 preceding and current row)'
        over_all = f'over(partition by contractsymbol order by datetime(gatherdate))'

        q1 = f'''
            select 
            *, 
            stk_price - lag(stk_price, 1) {lags} as stk_price_chg,
            avg(stk_price) {moving_avg} as stk_price_avg_30d,
            avg(stk_price) {fast_moving_avg} as stk_price_avg_5d,
            lastprice - lag(lastprice, 1) {lags} as lastprice_chg,
            avg(lastprice) {moving_avg} as lastprice_avg_30d,
            avg(lastprice) {fast_moving_avg} as lastprice_avg_5d,
            100*((lastprice - lag(lastprice, 1) {lags}) / lag(lastprice, 1) {lags}) as pct_chg,
            impliedvolatility - lag(impliedvolatility, 1) {lags} as iv_chg,
            avg(impliedvolatility) {fast_moving_avg} as iv_avg_5d,
            avg(impliedvolatility) {moving_avg} as iv_avg_30d,
            avg(impliedvolatility) {over_all} as iv_avg_all,
            vol - lag(vol, 1) {lags} as vol_chg,
            oi - lag(oi, 1) {lags} as oi_chg,
            case when (oi - lag(oi, 1) {lags}) > lag(vol, 1) {lags} then 1 else 0 end as flag,
            case when (oi - lag(oi, 1) {lags}) > lag(vol, 1) {lags} then ((oi - lag(oi, 1) {lags}) - lag(vol, 1) {lags}) else 0 end as amnt
            from (t0)
            '''

        q = f'''
            with t0 as ({q0}), t1 as ({q1})
            select * from t1
            -- only get contracts with more than 3 entries
            -- where contractsymbol in (select contractsymbol from t1 group by contractsymbol having count(*) > 3)

            '''
        g = self.option_db_cursor.execute(q)
        gr = g.fetchall()
        df = pd.DataFrame(gr, columns = [x[0] for x in g.description])
        return df.rename(columns = {'oi':'openinterest', 'vol':'volume'})
    
    
    def _initialize_change_db(self, stock):
        """ Calculate the change Variables for the first time """
        df = self._calc_changes(stock)
        df.to_sql(stock, self.change_db, if_exists = 'replace', index = False)
        self.change_db.commit()
        return df 
        
    def _update_change_vars(self, stock):
        """ Update the contracts in the change variable db. 
            1. Calculate the Changes for the last 3 days
            2. Update the changes in the vol.db by appending the max date from the calculation.
        """
        df = self._calc_changes(stock, N = 3) # Comes From Option DB
        df = df[df.gatherdate == df.gatherdate.max()]
        if self._check_for_stock_in_change_db(stock) == True:
            ## Make sure that you are not adding duplicates into the db 
            q = f''' select max(datetime(gatherdate)) from {stock} '''
            md = self.change_db_cursor.execute(q).fetchall()[0][0]
            if md == df.gatherdate.max():
                print("Dupicates Found")
                # df = pd.read_sql(f'select * from {stock}', self.change_db)
                # df = df.drop_duplicates()
                # df.to_sql(stock, self.change_db, if_exists = 'replace', index = False)
                return None
            else:
                df.to_sql(stock, self.change_db, if_exists = 'append', index = False)
                self.change_db.commit()
        else:
            df.to_sql(stock, self.change_db, if_exists = 'replace', index = False)
            self.change_db.commit()
        
        
if __name__ == "__main__":
    connections = {
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
    
    connections = {
            ##### Price Report ###########################
            'daily_db': 'data/prices/stocks.db', 
            'intraday_db': 'data/prices/stocks_intraday.db',
            'ticker_path': 'data/stocks/tickers.json',
            ##### Price Report ###########################
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
    
    oc = ChangeVars(connections)
    d = oc._calc_changes('gme')
    print(d[d.gatherdate == d.gatherdate.max()])
    oc.close_connections()
    
    option_type = "Call"
    strike = 20
    expiry = "2024-06-21"
    
    # print(d[(d['type'] == option_type) & (d.strike == strike) & (d.expiry == expiry)])
    
    
