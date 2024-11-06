"""
Call Put Percentage Database Setup Script. 


"""

import sys
import pandas as pd 
import numpy as np 
import yfinance as yf 
import datetime as dt 
from tqdm import tqdm

sys.path.append('/Users/jerald/Documents/Dir/Python/stocker')
from bin.options.optgd.db_connect import Connector
from bin.options.bsm.bsModel import bs_df


class CP(Connector):
    def __init__(self, connections):
        super().__init__(connections)
        
    def _custom_query_option_db(self, q, c):
        c.execute(q)
        d=  pd.DataFrame(c.fetchall(), columns = [desc[0] for desc in c.description])
        d['gatherdate'] = pd.to_datetime(d['gatherdate'])
        return d
    
    def _cp(self, stock, n = 60):
        gdate = self._max_dates(stock)
        q = f'''
        select 
        max(datetime(gatherdate)) as gatherdate,
        cast(sum(case when type = 'Call' then volume else 0 end) as int) as call_vol,
        cast(sum(case when type = 'Put' then volume else 0 end) as int) as put_vol,
        cast(sum(volume) as int) as total_vol,
        cast(sum(case when type = 'Call' then openinterest else 0 end) as int) as call_oi, 
        cast(sum(case when type = 'Put' then openinterest else 0 end) as int) as put_oi,
        cast(sum(openinterest) as int) as total_oi,
        cast(sum(case when type = 'Call' then cash else 0 end) as float) as call_prem, 
        cast(sum(case when type = 'Put' then cash else 0 end) as float) as put_prem,
        cast(sum(cash) as float) as total_prem, 
        cast(avg(case when type = 'Call' then impliedvolatility else 0 end) as float) as call_iv,
        cast(avg(case when type = 'Put' then impliedvolatility else 0 end) as float) as put_iv,
        cast(avg(case when stk_price / strike between 0.99 and 1.01 then impliedvolatility else 0 end) as float) as atm_iv, 
        cast(avg(case when stk_price / strike not between 0.99 and 1.01 then impliedvolatility else 0 end) as float) as otm_iv,
        cast(avg(case when type = 'Put' then ask - bid else 0 end) as float) as put_spread,
        cast(avg(case when type = 'Call' then ask - bid else 0 end) as float) as call_spread
        from "{stock}"
        where datetime(gatherdate) in ({gdate})
        and julianday(expiry) - julianday(gatherdate) < {n}
        group by date(gatherdate)
        order by gatherdate asc
        '''
        df = pd.read_sql_query(q, self.option_db, parse_dates = ['gatherdate']) 
        return df

    def _calculation(self, df):
        if 'gatherdate' in df.columns: 
            df['gatherdate'] = pd.to_datetime(df['gatherdate'])
            df = df.set_index('gatherdate')
        
        df['call_vol_pct'] = df['call_vol'] / df['total_vol']
        df['put_vol_pct'] = df['put_vol'] / df['total_vol']
        df['call_oi_pct'] = df['call_oi'] / df['total_oi']
        df['put_oi_pct'] = df['put_oi'] / df['total_oi']
        # lagges
        lag_df = df.diff(1)
        lag_df.columns = [f'{x}_chng' for x in lag_df.columns]
        df = pd.concat([df, lag_df], axis = 1).dropna()
        
        # Cast Columns with _chng to int, without columns that contain _pct 
        for col in df.columns:
            # replace inf with 1e-3
            df[col] = df[col].replace([np.inf, -np.inf], 1e-3)
            if 'oi|vol' not in col and '_pct' in col:
                df[col] = df[col].astype(float)
            if 'oi|vol' in col and '_pct' not in col:
                df[col] = df[col].astype(int)
        return df
    
    def _initialize_vol_db(self, stock):
        ''' Builds the table for the stock '''
        df = self._cp(stock)
        df = self._calculation(df)
        df.index = pd.to_datetime(df.index)
        df.to_sql(f'{stock}', self.vol_db, if_exists = 'replace')
        self.vol_db.commit()
        return df
    
    def _recent(self, stock):
        ''' Returns the last n rows of the stock table '''
        q = f'''
        select 
            datetime(gatherdate) as gatherdate, 
            call_vol, 
            put_vol, 
            total_vol, 
            call_oi, 
            put_oi, 
            total_oi, 
            call_prem, 
            put_prem, 
            total_prem,
            call_iv, 
            put_iv,
            atm_iv,
            otm_iv, 
            put_spread,
            call_spread
            
        from "{stock}"
        order by datetime(gatherdate) asc
        '''
        df = pd.read_sql_query(q, self.vol_db, parse_dates = ['gatherdate']).sort_values('gatherdate', ascending = True)
        return df 
    
    def update_cp(self, stock, new_chain):
        ''' Updates the table for stock with data from the new option chain '''
        new_chain['moneyness'] = new_chain['stk_price'] / new_chain['strike']
        
        chk = len(self._last_dates(stock)) > 3
        if chk != True:
            pass
        else:
            old_chain = self._recent(stock)
            calls = new_chain[new_chain['type'] == 'Call']
            puts = new_chain[new_chain['type'] == 'Put']
            newest_cp = pd.DataFrame({
            'gatherdate': calls['gatherdate'].max(),
            'call_vol': calls['volume'].sum(),
            'put_vol': puts['volume'].sum(),
            'total_vol': calls['volume'].sum() + puts['volume'].sum(),
            'call_oi': calls['openinterest'].sum(),
            'put_oi': puts['openinterest'].sum(),
            'total_oi': calls['openinterest'].sum() + puts['openinterest'].sum(),
            'call_prem': calls['cash'].sum(),
            'put_prem': puts['cash'].sum(),
            'total_prem': calls['cash'].sum() + puts['cash'].sum(), 
            'atm_iv': new_chain[(new_chain['moneyness'] >= 0.99) & (new_chain['moneyness'] <= 1.01)]['impliedvolatility'].mean(),
            'otm_iv': new_chain[(new_chain['moneyness'] < 0.99) | (new_chain['moneyness'] > 1.01)]['impliedvolatility'].mean(),
            'put_spread': (puts['ask'] - puts['bid']).mean(),
            'call_spread': (calls['ask'] - calls['bid']).mean()
            }, index =[0])
            ready = pd.concat([old_chain, newest_cp], axis = 0).reset_index(drop = True)
            add_on = self._calculation(ready).tail(1).reset_index()
            add_on.to_sql(f'{stock}', self.vol_db, if_exists = 'append', index = False)
            self.vol_db.commit()
            return pd.read_sql(f'select * from "{stock}"', self.vol_db)
    
    def __max_dates(self, stock):
        ''' Returns the max date in the database '''
        q0 = f'''
            select
            date(gatherdate) as gatherdate,
            max(datetime(gatherdate)) as maxdate
            from "{stock}"
            group by date(gatherdate)
        '''
        # df0 = pd.read_sql_query(q0, self.inactive_db)
        cursor = self.inactive_db.cursor()
        cursor.execute(q0)
        df0 = cursor.fetchall()
        df0 = pd.DataFrame(df0, columns = ['gatherdate', 'maxdate'])
        return ','.join([f"'{x}'" for x in df0['maxdate']])
    
    def get_cp_from_purged_db(self, stock, n = 60):
        gdate = self.__max_dates(stock)
        q = f'''
        select 
        max(datetime(gatherdate)) as gatherdate,
        cast(sum(case when type = 'Call' then volume else 0 end) as int) as call_vol,
        cast(sum(case when type = 'Put' then volume else 0 end) as int) as put_vol,
        cast(sum(volume) as int) as total_vol,
        cast(sum(case when type = 'Call' then openinterest else 0 end) as int) as call_oi, 
        cast(sum(case when type = 'Put' then openinterest else 0 end) as int) as put_oi,
        cast(sum(openinterest) as int) as total_oi,
        cast(sum(case when type = 'Call' then cash else 0 end) as float) as call_prem, 
        cast(sum(case when type = 'Put' then cash else 0 end) as float) as put_prem,
        cast(sum(cash) as float) as total_prem,
        cast(avg(case when type = 'Call' then impliedvolatility else 0 end) as float) as call_iv,
        cast(avg(case when type = 'Put' then impliedvolatility else 0 end) as float) as put_iv,
        cast(avg(case when stk_price / strike between 0.99 and 1.01 then impliedvolatility else 0 end) as float) as atm_iv, 
        cast(avg(case when stk_price / strike not between 0.99 and 1.01 then impliedvolatility else 0 end) as float) as otm_iv,
        cast(avg(case when type = 'Put' then ask - bid else 0 end) as float) as put_spread,
        cast(avg(case when type = 'Call' then ask - bid else 0 end) as float) as call_spread
        from "{stock}"
        where datetime(gatherdate) in ({gdate})
        and julianday(expiry) - julianday(gatherdate) < {n}
        group by date(gatherdate)
        order by gatherdate asc
        '''
        # df = pd.read_sql_query(q, self.inactive_db, parse_dates = ['gatherdate']) 
        cursor = self.inactive_db.cursor()
        cursor.execute(q)
        df = pd.DataFrame(cursor.fetchall(), columns = [desc[0] for desc in cursor.description])
        return self._calculation(df)
    
    def _intialized_cp(self, stock, n = 30):
        ''' Initializes the cp table '''
        try:
            old_df = self.get_cp_from_purged_db(stock, n = n)
        except:
            old_df = pd.DataFrame()
        current_df = self._calculation(self._cp(stock, n = n))
        new_df = pd.concat([old_df, current_df], axis = 0).reset_index()
        new_df.to_sql(f'{stock}', self.vol_db, if_exists = 'replace', index = False)
        self.vol_db.commit()

    def cp_query(self, stock, n = 30):
        try:
            old_df = self.get_cp_from_purged_db(stock, n = n)
        except:
            old_df = pd.DataFrame()
        current_df = self._calculation(self._cp(stock, n = n))
        new_df = pd.concat([old_df, current_df], axis = 0).reset_index()
        return new_df
    
   
        
    
    
if __name__ == "__main__":
    # print("How much longer are you going to wait to demand the best for yourself?") 
    print("(10.4) Spiritual Intelligence, Knowledge, freedom from false perception, compassion, trufhfullness, control of the senses, control of the mind, happiness, unhappiness, birth, death, fear and fearlessness, nonviolence, equanimity,  contentment, austerity, charity, fame, infamy; all these variegated diverse qualities of all living entities originate from Me alone.")
    connections = {
                ##### Price Report ###########################
                'daily_db': 'data/prices/stocks.db', 
                'intraday_db': 'data/prices/stocks_intraday.db',
                'ticker_path': 'data/stocks/tickers.json',
                ################################################
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
    cp = CP(connections)
    st = cp.stocks['equities']
    st = ['^RUT']
    for i in tqdm(st):cp._intialized_cp(i)
    # df = cp.daily_option_stats('spy', write = True).set_index('gatherdate')
    # cols = list(df.filter(regex='straddle|exp|iv'))
    # print(df[cols])
    # df = cp.cp_query('spy')
    # print(df["2023-06-01":])
    # print('\n\n')
    cp.close_connections()
