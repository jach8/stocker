# improved_cp.py

import sys
# Set Path 

from typing import Any, Dict, List, Tuple, Optional
import pandas as pd 
import numpy as np 
import yfinance as yf 
import datetime as dt 
from tqdm import tqdm
import sqlite3 as sql
import logging
from bin.options.optgd.db_connect import Connector
from models.bsm.bsModel import bs_df

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class CP(Connector):
    """
    Engineering Features for each stock in the database. Using the option chain, the following script aims to extract critical information about the stock for that date. 
    Each entry in the table has a Datetime index named 'gatherdate', and is in the format %Y-%m-%d %H:%M:%S sometimes it is %Y-%m-%dT%H:%M:%S.
    
    The following features are derived from the option chain: 
        1. 'call_vol': The total volume of call options traded for that day.
        2. 'put_vol': The total volume of put options traded for that day.
        3. 'total_vol': The total volume of options traded for that day.
        4. 'call_oi': The total open interest of call options for that day.
        5. 'put_oi': The total open interest of put options for that day.
        6. 'total_oi': The total open interest of options for that day.
        7. 'call_prem': The total premium of call options for that day.
        8. 'put_prem': The total premium of put options for that day.
        9. 'total_prem': The total premium of options for that day.
        10. 'call_iv': The average implied volatility of call options for that day.
        11. 'put_iv': The average implied volatility of put options for that day.
        12. 'atm_iv': The average implied volatility of options that are at the money for that day.
        13. 'otm_iv': The average implied volatility of options that are out of the money for that day.
        14. 'put_spread': The average spread (ask - bid) of put options for that day.
        15. 'call_spread': The average spread (ask - bid) of call options for that day.
    """

    def __init__(self, connections: Dict[str, Any]):
        """ Import Connections """
        super().__init__(connections)
        try:
            self.dates_db = sql.connect(connections['dates_db'])
        except sql.Error as e:
            logging.error(f"Failed to connect to dates_db: {e}", exc_info=True)
            raise

    def __custom_query_option_db(self, q: str, connection: sql.Connection) -> pd.DataFrame:
        """ 
        Helper function to run custom queries on the option database 
            args: 
                q: str: query 
                connection: sql.Connection: connection to the database
            returns:
                pd.DataFrame: DataFrame of the query results
        """
        try:
            c = connection.cursor()
            c.execute(q)
            d = pd.DataFrame(c.fetchall(), columns=[desc[0] for desc in c.description])
            d['gatherdate'] = pd.to_datetime(d['gatherdate'])
            return d
        except sql.Error as e:
            logging.error(f"Error executing custom query '{q[:50]}...': {e}", exc_info=True)
            raise

    def _cp(self, stock: str, n: int = 300) -> pd.DataFrame:
        q = f'''
        SELECT 
        MAX(datetime(gatherdate)) AS gatherdate,
        CAST(SUM(CASE WHEN type = 'Call' THEN volume ELSE 0 END) AS INT) AS call_vol,
        CAST(SUM(CASE WHEN type = 'Put' THEN volume ELSE 0 END) AS INT) AS put_vol,
        CAST(SUM(volume) AS INT) AS total_vol,
        CAST(SUM(CASE WHEN type = 'Call' THEN openinterest ELSE 0 END) AS INT) AS call_oi, 
        CAST(SUM(CASE WHEN type = 'Put' THEN openinterest ELSE 0 END) AS INT) AS put_oi,
        CAST(SUM(openinterest) AS INT) AS total_oi,
        CAST(SUM(CASE WHEN type = 'Call' THEN cash ELSE 0 END) AS FLOAT) AS call_prem, 
        CAST(SUM(CASE WHEN type = 'Put' THEN cash ELSE 0 END) AS FLOAT) AS put_prem,
        CAST(SUM(cash) AS FLOAT) AS total_prem, 
        CAST(AVG(CASE WHEN type = 'Call' THEN impliedvolatility ELSE 0 END) AS FLOAT) AS call_iv,
        CAST(AVG(CASE WHEN type = 'Put' THEN impliedvolatility ELSE 0 END) AS FLOAT) AS put_iv,
        CAST(AVG(CASE WHEN stk_price / strike BETWEEN 0.99 AND 1.01 THEN impliedvolatility ELSE 0 END) AS FLOAT) AS atm_iv, 
        CAST(AVG(CASE WHEN stk_price / strike NOT BETWEEN 0.99 AND 1.01 THEN impliedvolatility ELSE 0 END) AS FLOAT) AS otm_iv,
        CAST(AVG(CASE WHEN type = 'Put' THEN ask - bid ELSE 0 END) AS FLOAT) AS put_spread,
        CAST(AVG(CASE WHEN type = 'Call' THEN ask - bid ELSE 0 END) AS FLOAT) AS call_spread
        FROM {stock}
        GROUP BY date(gatherdate)
        ORDER BY gatherdate ASC
        '''
        try:
            logging.info(f"Running _cp for {stock.upper()}")
            return self.__custom_query_option_db(q, self.option_db)
        except Exception as e:
            logging.error(f"Error in _cp for {stock}: {e}", exc_info=True)
            raise

    def _calculation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the change in the features observed above. 
            args:
                df: pd.DataFrame: DataFrame of the option chain
            returns:
                pd.DataFrame: DataFrame with the change variables appended to the original dataframe 
        """
        try:
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
            df = pd.concat([df, lag_df], axis=1).dropna()
            
            # Cast Columns with _chng to int, without columns that contain _pct 
            for col in df.columns:
                # replace inf with 1e-3
                df[col] = df[col].replace([np.inf, -np.inf], 1e-3)
                if 'oi|vol' not in col and '_pct' in col:
                    df[col] = df[col].astype(float)
                if 'oi|vol' in col and '_pct' not in col:
                    df[col] = df[col].astype(int)
            return df
        except Exception as e:
            logging.error(f"Error in _calculation: {e}", exc_info=True)
            raise

    def _initialize_vol_db(self, stock: str) -> pd.DataFrame:
        ''' Builds the table for the stock 
        
        args:
            stock: str: stock symbol
        returns:
            pd.DataFrame: DataFrame of the stock table
        
        '''
        try:
            df = self._cp(stock)
            df = self._calculation(df)
            df.index = pd.to_datetime(df.index)
            df.to_sql(f'{stock}', self.vol_db, if_exists='replace', index=False)
            self.vol_db.commit()
            logging.info(f"Initialized vol_db for {stock}")
            return df
        except Exception as e:
            logging.error(f"Error initializing vol_db for {stock}: {e}", exc_info=True)
            self.vol_db.rollback()
            raise

    def _recent(self, stock: str) -> pd.DataFrame:
        ''' Returns the last n rows of the stock table '''
        q = f'''
        SELECT 
            datetime(gatherdate) AS gatherdate, 
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
        FROM {stock}
        ORDER BY datetime(gatherdate) ASC
        '''
        try:
            df = pd.read_sql_query(q, self.vol_db, parse_dates=['gatherdate']).sort_values('gatherdate', ascending=True)
            logging.info(f"Fetched recent data for {stock}")
            return df 
        except sql.Error as e:
            logging.error(f"Error fetching recent data for {stock}: {e}", exc_info=True)
            raise

    def _last_dates(self, stock: str, N: int = 5) -> np.ndarray:
        ''' Return the last dates for each day for a stock in the db '''
        q = f'''
        SELECT DISTINCT
            MAX(datetime(gatherdate)) OVER (PARTITION BY date(gatherdate)) AS gatherdate
        FROM {stock}
        ORDER BY gatherdate DESC LIMIT ?
        '''
        try:
            cursor = self.vol_db.cursor()
            cursor.execute(q, (N,))
            df = pd.DataFrame(cursor.fetchall(), columns=['gatherdate'])
            return df['gatherdate'].unique()
        except sql.Error as e:
            logging.error(f"Error fetching last dates for {stock}: {e}", exc_info=True)
            raise

    def update_cp(self, stock: str, new_chain: pd.DataFrame) -> Optional[pd.DataFrame]:
        ''' Updates the table for stock with data from the new option chain '''
        try:
            new_chain['moneyness'] = new_chain['stk_price'] / new_chain['strike']
            
            chk = len(self._last_dates(stock)) > 3
            if not chk:
                logging.warning(f"Not enough historical data for {stock}. Skipping update.")
                return None
            else:
                old_chain = self._recent(stock)
                calls = new_chain[new_chain['type'] == 'Call']
                puts = new_chain[new_chain['type'] == 'Put']
                newest_cp = pd.DataFrame({
                'gatherdate': [calls['gatherdate'].max()],
                'call_vol': [calls['volume'].sum()],
                'put_vol': [puts['volume'].sum()],
                'total_vol': [calls['volume'].sum() + puts['volume'].sum()],
                'call_oi': [calls['openinterest'].sum()],
                'put_oi': [puts['openinterest'].sum()],
                'total_oi': [calls['openinterest'].sum() + puts['openinterest'].sum()],
                'call_prem': [calls['cash'].sum()],
                'put_prem': [puts['cash'].sum()],
                'total_prem': [calls['cash'].sum() + puts['cash'].sum()], 
                'atm_iv': [new_chain[(new_chain['moneyness'] >= 0.99) & (new_chain['moneyness'] <= 1.01)]['impliedvolatility'].mean()],
                'otm_iv': [new_chain[(new_chain['moneyness'] < 0.99) | (new_chain['moneyness'] > 1.01)]['impliedvolatility'].mean()],
                'put_spread': [(puts['ask'] - puts['bid']).mean()],
                'call_spread': [(calls['ask'] - calls['bid']).mean()]
                })
                ready = pd.concat([old_chain, newest_cp], axis=0, ignore_index=True)
                add_on = self._calculation(ready).tail(1).reset_index(drop=True)
                add_on.to_sql(f'{stock}', self.vol_db, if_exists='append', index=False)
                self.vol_db.commit()
                logging.info(f"Updated {stock} in vol_db")
                return pd.read_sql(f'select * from {stock}', self.vol_db)
        except Exception as e:
            logging.error(f"Error updating {stock}: {e}", exc_info=True)
            self.vol_db.rollback()
            raise

    def __max_dates(self, stock: str) -> str:
        ''' Returns the max date in the database '''
        q0 = f'''
            SELECT
            date(gatherdate) AS gatherdate,
            MAX(datetime(gatherdate)) AS maxdate
            FROM {stock}
            GROUP BY date(gatherdate)
        '''
        try:
            cursor = self.inactive_db.cursor()
            cursor.execute(q0)
            df0 = pd.DataFrame(cursor.fetchall(), columns=['gatherdate', 'maxdate'])
            return ','.join([f"'{x}'" for x in df0['maxdate']])
        except sql.Error as e:
            logging.error(f"Error fetching max dates for {stock}: {e}", exc_info=True)
            raise

    def get_cp_from_purged_db(self, stock: str, n: int = 300) -> pd.DataFrame:
        try:
            # gdate = self.__max_dates(stock)
            q = f'''
            SELECT 
            MAX(datetime(gatherdate)) AS gatherdate,
            CAST(SUM(CASE WHEN type = 'Call' THEN volume ELSE 0 END) AS INT) AS call_vol,
            CAST(SUM(CASE WHEN type = 'Put' THEN volume ELSE 0 END) AS INT) AS put_vol,
            CAST(SUM(volume) AS INT) AS total_vol,
            CAST(SUM(CASE WHEN type = 'Call' THEN openinterest ELSE 0 END) AS INT) AS call_oi, 
            CAST(SUM(CASE WHEN type = 'Put' THEN openinterest ELSE 0 END) AS INT) AS put_oi,
            CAST(SUM(openinterest) AS INT) AS total_oi,
            CAST(SUM(CASE WHEN type = 'Call' THEN cash ELSE 0 END) AS FLOAT) AS call_prem, 
            CAST(SUM(CASE WHEN type = 'Put' THEN cash ELSE 0 END) AS FLOAT) AS put_prem,
            CAST(SUM(cash) AS FLOAT) AS total_prem, 
            CAST(AVG(CASE WHEN type = 'Call' THEN impliedvolatility ELSE 0 END) AS FLOAT) AS call_iv,
            CAST(AVG(CASE WHEN type = 'Put' THEN impliedvolatility ELSE 0 END) AS FLOAT) AS put_iv,
            CAST(AVG(CASE WHEN stk_price / strike BETWEEN 0.99 AND 1.01 THEN impliedvolatility ELSE 0 END) AS FLOAT) AS atm_iv, 
            CAST(AVG(CASE WHEN stk_price / strike NOT BETWEEN 0.99 AND 1.01 THEN impliedvolatility ELSE 0 END) AS FLOAT) AS otm_iv,
            CAST(AVG(CASE WHEN type = 'Put' THEN ask - bid ELSE 0 END) AS FLOAT) AS put_spread,
            CAST(AVG(CASE WHEN type = 'Call' THEN ask - bid ELSE 0 END) AS FLOAT) AS call_spread
            FROM {stock}
            GROUP BY date(gatherdate)
            ORDER BY gatherdate ASC
            '''
            df = self.__custom_query_option_db(q, self.inactive_db, (n,))
            return self._calculation(df)
        except Exception as e:
            logging.error(f"Error getting CP from purged db for {stock}: {e}", exc_info=True)
            raise

    def _intialized_cp(self, stock: str, n: int = 30) -> None:
        ''' Initializes the cp table '''
        try:
            old_df = self.get_cp_from_purged_db(stock, n=n)
        except:
            old_df = pd.DataFrame()
        try:
            current_df = self._calculation(self._cp(stock, n=n))
            new_df = pd.concat([old_df, current_df], axis=0).reset_index().drop_duplicates()
            new_df.to_sql(f'{stock}', self.vol_db, if_exists='append', index=False)
            self.vol_db.commit()
            logging.info(f"Initialized CP for {stock}")
        except Exception as e:
            logging.error(f"Error initializing CP for {stock}: {e}", exc_info=True)
            self.vol_db.rollback()
            raise

    def cp_query(self, stock: str, n: int = 30) -> pd.DataFrame:
        try:
            old_df = self.get_cp_from_purged_db(stock, n=n)
        except:
            old_df = pd.DataFrame()
        try:
            current_df = self._calculation(self._cp(stock, n=n))
            new_df = pd.concat([old_df, current_df], axis=0).reset_index()
            return new_df
        except Exception as e:
            logging.error(f"Error in CP query for {stock}: {e}", exc_info=True)
            raise

if __name__ == "__main__":
    print("(10.4) Spiritual Intelligence, Knowledge, freedom from false perception, compassion, trufhfullness, control of the senses, control of the mind, happiness, unhappiness, birth, death, fear and fearlessness, nonviolence, equanimity,  contentment, austerity, charity, fame, infamy; all these variegated diverse qualities of all living entities originate from Me alone.")
    import sys 
    # Set Path 
    from bin.main import get_path
    connections = get_path()
    print()
    cp = CP(connections)
    
    try:
        current= pd.read_sql('select * from ibm', cp.vol_db, parse_dates=['gatherdate'], index_col='gatherdate').sort_index()
        print(current)
        print(current.drop_duplicates())
        print()
        print(current.groupby(current.index.date).sum())
        print()
        print(cp._calculation(cp._cp('ibm')))
    except Exception as e:
        logging.error(f"Error in main script: {e}", exc_info=True)
        raise e
    finally:
        cp.close_connections()