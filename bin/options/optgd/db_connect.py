""" 
Connect to all databases 
    - Option database 
    - Option Statistics Database
    - Tickers 
    - Option Change DB ** This is updated, tracking the change in each individual contracts. 
    
Contains some utility functions for: 
    1. Inserting New Data into the database, 
    2. Checking for the stock in the option database
    3. Getting the last 5 days of option data for a stock. ** To be used in tracking the change in each contract. 
    4. Getting the last dates for each day for a stock in the db. ** To be used in tracking the change in each contract.

"""


import sqlite3 as sql 
import numpy as np 
import pandas as pd 
import datetime as dt 
import re
import time
import json 
import os 

def check_connection(path):
    if os.path.exists(path):
        return True
    else:
        return False

class Connector: 
    """ 
    Database Connector for Options Data
    """
    def __init__(self, connections):
        self.execution_start_time = time.time()
        try:
            # Add stocks, Keys: ['all_stocks', 'bonds','etf', 'equities', 'market', 'mag8']
            self.stocks = json.load(open(connections['ticker_path'], 'r'))     
            self.all_stocks = self.stocks['all_stocks']
            
            # Add Connections 
            self.path_dict = connections
            self.option_db = sql.connect('file:' + connections['option_db'] + '?mode=ro', uri = True)
            self.option_db_cursor = self.option_db.cursor()
            
            self.write_option_db = sql.connect(connections['option_db'])
            self.write_option_db_cursor = self.write_option_db.cursor()
            
            self.change_db = sql.connect(connections['change_db'])
            self.change_db_cursor = self.change_db.cursor()
            
            self.vol_db = sql.connect(connections['vol_db'])
            self.stats_db = sql.connect(connections['stats_db'])
            self.tracking_db = sql.connect(connections['tracking_db'])
            self.tracking_values_db = sql.connect(connections['tracking_values_db'])
            self.backup = sql.connect(connections['backup_db'])
            
            
            self.inactive_db = sql.connect(connections['inactive_db'])
            self.inactive_db_cursor = self.inactive_db.cursor()

            print("Options db Connected: {}".format(dt.datetime.now()))
        
        except Exception as e:
            print("Error: ", e)
            # Print the last defined variable
            print(self.__dict__)
            raise Exception("Error: ", e)

        
        
        
    def _purge_inactive(self, stock):
        """
        Purge Inactive Contracts from the Option_db database. 
            - Save them in the inactive_db so that we can use them for tracking. 
        """
        exp_q = f''' select * from "{stock}" where date(expiry) < date('now') '''
        exp = pd.read_sql_query(exp_q, self.option_db)
        if exp.shape[0] > 0:
            contracts = ','.join([f'"{x}"' for x in exp.contractsymbol])
            change_db_q = f''' select * from "{stock}" where contractsymbol in ({contracts}) '''
            cdb = pd.read_sql_query(change_db_q, self.change_db)
            
            if self._check_inactive_db_for_stock(stock):
                print("EXISTING TABLE",len(exp), len(cdb))
                exp.to_sql(stock, self.inactive_db, if_exists='append', index=False)
                cdb.to_sql(stock + "_change", self.inactive_db, if_exists='append', index=False)
            else:
                print("NEW TABLE:",len(exp), len(cdb))
                exp.to_sql(stock, self.inactive_db, if_exists='replace', index=False)
                cdb.to_sql(stock + "_change", self.inactive_db, if_exists='replace', index=False)
            
            # Purge from Option DB 
            self.write_option_db_cursor.execute(f'delete from "{stock}" where date(expiry) < date("now")')
            self.write_option_db.commit()
            
            # Purge from Change DB
            self.change_db_cursor.execute(f'delete from "{stock}" where contractsymbol in ({contracts})')
            self.change_db.commit()
        
    def close_connections(self):
        for i in [self.option_db, self.write_option_db, self.change_db, self.vol_db, self.stats_db, self.tracking_db, self.tracking_values_db, self.backup]:
            i.close()
        end_time = time.time()
        runtime_min = (end_time - self.execution_start_time) / 60
        print("Connections Closed {}\nTotal Runtime: {:.2f} min".format(dt.datetime.now(), runtime_min))
        print()
        return None

    def clear_tables(self):
        ''' Save a log and clear the Unusual Activity, exp move, and high percentage tables. '''
        # save backup file 
        self.stats_db.backup(self.backup)
        c = self.stats_db.cursor()
        tables = [i[0] for i in c.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]
        for i in tables:
            c.execute(f'drop table {i}')
            self.stats_db.commit()
        return None
    
    def _check_for_stock_in_option_db(self, stock):
        """ Check if the stock is in the option database """
        check = self.option_db_cursor.execute(f'select name from sqlite_master where type = "table" and name = "{stock}"').fetchall()
        if check != []:
            return False
        else:
            return True
        
    def _check_for_stock_in_change_db(self, stock):
        """ Check if the stock is in the change database """
        check = self.change_db_cursor.execute(f'select name from sqlite_master where type = "table" and name = "{stock}"').fetchall()
        if check != []:
            return False
        else:
            return True
    
    def _check_tracking_chain_for_stock(self, stock):
        """ Check if the stock has an existing table in the tracking db and if the table is empty
            Returns:
                - True if the stock is in the tracking db 
                - False if the stock is not in the tracking db
        """
        q = f'select name from sqlite_master where type="table" and name="{stock}"'
        cursor = self.tracking_db.cursor()
        res = cursor.execute(q).fetchall()
        if res != []:
            return True
        else:
            return False
    
    def _check_tracking_values_for_stock(self, stock):
        """ Read in the Contracts we are tracking
            Returns:
                - False if the stock is not in the tracking values db 
                - True if the stock is in the tracking values db
        """
        valid_tickers = pd.read_sql('select * from sqlite_master where type="table"', self.tracking_values_db)
        if stock not in valid_tickers.name.values:
            return False
        else:
            q = f'select count(*) from "{stock}"'
            if self.tracking_values_db.execute(q).fetchone()[0] == 0:
                return False
            else:
                return True
        
    def _check_inactive_db_for_stock(self, stock): 
        """ Check if the stock is in the inactive database """
        valid = pd.read_sql('select * from sqlite_master where type="table"', self.inactive_db)
        if stock in valid.name.values:
            return True
        else:
            return False
        
    
    def _recent_option_chain(self, stock):
        """ Return the last 5 days of option data """
        if self._check_for_stock_in_option_db(stock):
            print(f'${stock.upper()} not in the database')
            return None
        else:
            dates = self._last_dates(stock)
            dates = ','.join([f'"{x}"' for x in dates])
            q = f''' select * from "{stock}" where datetime(gatherdate) in ({dates}) '''
            return pd.read_sql_query(q, self.option_db)
    
    
    def _last_dates(self, stock, N = 5):
        """ Return the last dates for each day for a stock in the db """
        q = f'''
        select 
            distinct
            last_value (datetime(gatherdate)) over 
                (partition by date(gatherdate) rows between 
                unbounded preceding and unbounded following) 
            as gatherdate
        from "{stock}"
            where date(gatherdate) > date("2022-11-15")
        '''
        cursor = self.option_db.cursor()
        out = cursor.execute(q).fetchall()
        out = [x[0] for x in out]
        return out[-N:]
    
    def _max_dates(self, stock):
        ''' Returns the max date in the database '''
        q0 = f'''
            select
            date(gatherdate) as gatherdate,
            max(datetime(gatherdate)) as maxdate
            from "{stock}"
            group by date(gatherdate)
        '''
        df0 = pd.read_sql_query(q0, self.option_db)
        return ','.join([f"'{x}'" for x in df0['maxdate']])

    def describe_option(self, y):
        ''' Given an option contract symbol, using regular expressions to return the stock, expiration date, contract type, and strike price. '''
        valid = re.compile(r'(?P<stock>[A-Z]+)(?P<expiry>[0-9]+)(?P<type>[C|P])(?P<strike>[0-9]+)')
        stock = valid.match(y).group('stock')
        expiration = dt.datetime.strptime(valid.match(y).group('expiry'), "%y%m%d")
        conttype = valid.match(y).group('type')
        strike = float(valid.match(y).group('strike')) / 1000
        return ("$"+stock, conttype, float(strike), expiration.strftime('%m/%d/%y'))
    
        
        
        
if __name__ == "__main__":
    print("True Humility is not thinking less of yourself; It is thinking of yourself less.")

    connections = {
            'inactive_db': 'bin/pipe/log/inactive.db',
            'backup_db': 'bin/pipe/log/backup.db',
            'tracking_values_db': 'data/options/bin/pipe/test_data/tracking_values.db',
            'tracking_db': 'data/options/bin/pipe/test_data/tracking.db',
            'stats_db': 'bin/pipe/test_data/stats.db',
            'vol_db': 'bin/pipe/test_data/vol.db',
            'change_db': 'bin//pipe/test_data/option_change.db', 
            'option_db': 'bin/pipe/test_data/test.db', 
            'testing_option_db': 'bin/pipe/test_data/test.db',
            'options_stat': 'bin/pipe/test_data/options_stat.db',
            'ticker_path': 'data/stocks/tickers.json'
            }    
    conn = Connector(connections)
    conn._purge_inactive('spy')
    conn.close_connections()        
            
            
