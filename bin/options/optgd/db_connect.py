"""
Database connector for options data using connection pooling.

This module provides a thread-safe database connector that leverages connection pooling
for efficient database access and resource management.
"""

import json
import time
import datetime as dt
import pandas as pd
from typing import Dict, Optional

from bin.utils.connection_pool import get_pool

class Connector:
    def __init__(self, connections: Dict[str, str]):
        """
        Database Connector for Options Data.
        
        Args:
            connections (dict): Dictionary of the paths to the databases.
                Example:
                    {
                        'option_db': 'Path to the option database',
                        'change_db': 'Path to the option change database',
                        'vol_db': 'Path to the volume database',
                        'stats_db': 'Path to the statistics database',
                        'tracking_db': 'Path to the tracking database',
                        'tracking_values_db': 'Path to the tracking values database',
                        'backup_db': 'Path to the backup database',
                        'inactive_db': 'Path to the inactive database',
                        'ticker_path': 'Path to the ticker json file'
                    }
                    
        Attributes:
            stocks: Dictionary of the stocks.
            path_dict: Dictionary of database paths.
            pool: Connection pool instance.
        """
        self.execution_start_time = time.time()
        self.path_dict = connections
        self.pool = get_pool()
        
        try:
            # Load stocks configuration
            with open(connections['ticker_path'], 'r') as f:
                self.stocks = json.load(f)
                
            print(f"Options db Connected: {dt.datetime.now()}")
            
        except Exception as e:
            error_msg = f"Connection Failed: {str(e)}"
            print(error_msg)
            raise Exception(error_msg)

    def __check_inactive_db_for_stock(self, stock: str) -> bool:
        """
        Check if the stock is in the inactive database.
        
        Args:
            stock (str): Stock Ticker Symbol
        
        Returns:
            bool: True if the stock is in the database, False if not.
        """
        query = f"""
        SELECT EXISTS(
            SELECT 1 FROM sqlite_master 
            WHERE type='table' AND name='{stock}'
        )
        """
        with self.pool.get_connection('inactive') as conn:
            cursor = conn.cursor()
            valid = cursor.execute(query).fetchone()[0]
            return bool(valid)

    def __purge_inactive(self, stock: str) -> None:
        """
        Purge Inactive Contracts from the Option_db database.
            - Save them in the inactive_db so that we can use them for tracking.
        
        Args:
            stock (str): Stock Ticker Symbol
        """
        # Get expired contracts from options DB
        exp_query = f"SELECT * FROM {stock} WHERE date(expiry) < date('now')"
        
        with self.pool.get_connection('options') as conn:
            cursor = conn.cursor()
            exp = cursor.execute(exp_query).fetchall()
            columns = [x[0] for x in cursor.description]
            exp_df = pd.DataFrame(exp, columns=columns)
        
        if exp_df.empty:
            return
            
        # Get change data for expired contracts
        contracts = ','.join([f'"{x}"' for x in exp_df.contractsymbol])
        change_query = f"SELECT * FROM {stock} WHERE contractsymbol IN ({contracts})"
        
        with self.pool.get_connection('changes') as conn:
            change_df = pd.read_sql_query(change_query, conn)
        
        # Save to inactive DB
        with self.pool.get_connection('inactive') as conn:
            if_exists = 'append' if self.__check_inactive_db_for_stock(stock) else 'replace'
            exp_df.to_sql(stock, conn, if_exists=if_exists, index=False)
            change_df.to_sql(f"{stock}_change", conn, if_exists=if_exists, index=False)
            
            print(f"{'EXISTING' if if_exists == 'append' else 'NEW'} TABLE: "
                  f"{len(exp_df)}, {len(change_df)}")
        
        # Delete from options DB
        with self.pool.get_connection('options') as conn:
            cursor = conn.cursor()
            cursor.execute(f'DELETE FROM {stock} WHERE date(expiry) < date("now")')
            conn.commit()
        
        # Delete from change DB
        with self.pool.get_connection('changes') as conn:
            cursor = conn.cursor()
            cursor.execute(f'DELETE FROM {stock} WHERE contractsymbol IN ({contracts})')
            conn.commit()

    def close_connections(self) -> None:
        """
        Close all connections in the pool.
        """
        if self.pool:
            self.pool.close_all()
            
        end_time = time.time()
        runtime_min = (end_time - self.execution_start_time) / 60
        print(f"Connections Closed {dt.datetime.now()}")
        print(f"Total Runtime: {runtime_min:.2f} min\n")

if __name__ == "__main__":
    print("True Humility is not thinking less of yourself; It is thinking of yourself less.")
    connections = get_path()  # This should be imported or defined elsewhere
    conn = Connector(connections)
    # conn.__purge_inactive('spy')
    conn.close_connections()