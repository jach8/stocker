"""
Enhanced main module with connection pooling and improved database operations
"""

import json
import pandas as pd 
import numpy as np 
import sqlite3 as sql
import datetime as dt
import logging
import time
from typing import Dict, Optional, List
from pathlib import Path
from tqdm import tqdm 
from dotenv import load_dotenv
import sys
import os

from bin.utils.connection_pool import get_pool, ConnectionPool
from bin.price.db_connect import Prices
from bin.options.manage_all import Manager as Optionsdb
from bin.earnings.get_earnings import Earnings 
from bin.price.perf.report import perf as performance
from bin.alerts.options_alerts import Notifications
from bin.signals.plays.ling import Scanner
from bin.signals.plays.dxp import dxp 
from bin.utils.Initialize import Initialize
from bin.utils.add_stocks import add_stock

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def setup_logger():
    """Configure logging with file and console handlers"""
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        'logs/pipeline.log',
        maxBytes=10*1024*1024,
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

class Manager:
    """Base manager class with connection pooling"""
    
    def __init__(self, connections: Optional[Dict[str, str]] = None):
        try:
            # Initialize connection pool
            self.pool = get_pool()
            
            # Load connections
            if isinstance(connections, str):
                connections = get_path(connections)
            elif connections is None:
                connections = get_path()
            elif not isinstance(connections, dict):
                raise ValueError('Connections must be a dictionary')
                
            # Store connection paths
            self.connection_paths = connections
            
            # Initialize components with connection paths
            self.Optionsdb = Optionsdb(connections)
            self.Pricedb = Prices(connections)
            self.performance = performance(connections)
            self.Notifications = Notifications(connections)
            self.Scanner = Scanner(connections)
            self.dxp = dxp(connections)
            
            logger.info("Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Manager initialization failed: {str(e)}")
            self.close_connection()
            raise
            
    def close_connection(self):
        """Close all database connections"""
        try:
            if hasattr(self, 'Optionsdb'):
                self.Optionsdb.close_connections()
            if hasattr(self, 'Pricedb'):
                self.Pricedb.close_connections()
            if hasattr(self, 'pool'):
                self.pool.close_all()
        except Exception as e:
            logger.error(f"Error closing connections: {str(e)}")
            raise
            
    def addStock(self, stock: str):
        """Add a new stock to the database"""
        try:
            with self.pool.get_connection('stock_names') as conn:
                add_stock(conn, self.connection_paths['ticker_path'], stock)
                logger.info(f"Successfully added stock: {stock}")
        except Exception as e:
            logger.error(f"Failed to add stock {stock}: {str(e)}")
            raise

class Pipeline(Manager):
    """Pipeline for processing options and stock data"""
    
    def __init__(self, connections: Optional[Dict[str, str]] = None):
        """Initialize the pipeline"""
        super().__init__(connections)
        self.begin = '\033[92m'
        self.endoc = '\033[0m'
        
    def update_options(self):
        """Update options data for all stocks"""
        try:
            out = []
            stocks = self.Optionsdb.stocks['all_stocks']
            pbar = tqdm(stocks, desc="Options Data")
            
            for stock in pbar:
                try:
                    pbar.set_description(f"Options Data {self.begin}${stock.upper()}{self.endoc}")
                    
                    # Get new option chain
                    new_chain = self.Optionsdb.insert_new_chain(stock)
                    if new_chain is None:
                        logger.warning(f"No new chain data for {stock}")
                        continue
                        
                    # Initialize and update databases
                    self.Optionsdb._initialize_change_db(stock)
                    self.Optionsdb._intialized_cp(stock)
                    self.Notifications.notifications(stock, n=10)
                    
                    # Rate limiting
                    time.sleep(2.5)
                    
                except Exception as e:
                    logger.error(f"Error processing {stock}: {str(e)}")
                    continue
                    
            # Process aggregate data
            self.Optionsdb._all_cp()
            self.Optionsdb._init_em_tables()
            self.Scanner.scan_contracts()
            
            logger.info("Options update completed successfully")
            
        except Exception as e:
            logger.error(f"Options update failed: {str(e)}")
            raise
            
    def workflow(self):
        """Run the standard workflow"""
        try:
            # Gather option stats
            self.Optionsdb._all_cp()
            
            # Process expected move data
            self.Optionsdb._init_em_tables()
            
            # Scan for plays
            self.Scanner.scan_contracts()
            
            logger.info("Workflow completed successfully")
            
        except Exception as e:
            logger.error(f"Workflow failed: {str(e)}")
            raise
            
    def update_stock_prices(self):
        """Update stock price data"""
        try:
            self.Pricedb.update_stock_prices()
            self.performance.show_performance()
            logger.info("Stock prices updated successfully")
            
        except Exception as e:
            logger.error(f"Stock price update failed: {str(e)}")
            raise
            
    def view_notifications(self):
        """View all notifications"""
        try:
            self.Notifications.iterator()
        except Exception as e:
            logger.error(f"Failed to view notifications: {str(e)}")
            raise
            
    def master_run(self):
        """Run the complete pipeline"""
        try:
            logger.info("Starting master run")
            
            # Update stock prices
            self.update_stock_prices()
            
            # Update options data
            self.update_options()
            
            # Run workflow
            self.workflow()
            
            logger.info("Master run completed successfully")
            
        except Exception as e:
            logger.error(f"Master run failed: {str(e)}")
            raise
        finally:
            self.close_connection()

def get_path(pre: str = '') -> Dict[str, str]:
    """Get database paths from environment or defaults"""
    try:
        load_dotenv()
        
        connections = {
            # Bonds Data
            'bonds_db': os.getenv('BONDS_DB', f'{pre}data/bonds/bonds.db'),
            
            # Price Data
            'daily_db': os.getenv('DAILY_DB', f'{pre}data/prices/stocks.db'),
            'intraday_db': os.getenv('INTRADAY_DB', f'{pre}data/prices/stocks_intraday.db'),
            'ticker_path': os.getenv('TICKER_PATH', f'{pre}data/stocks/tickers.json'),
            
            # Options Data
            'inactive_db': os.getenv('INACTIVE_DB', f'{pre}data/options/log/inactive.db'),
            'backup_db': os.getenv('BACKUP_DB', f'{pre}data/options/log/backup.db'),
            'tracking_values_db': os.getenv('TRACKING_VALUES_DB', f'{pre}data/options/tracking_values.db'),
            'tracking_db': os.getenv('TRACKING_DB', f'{pre}data/options/tracking.db'),
            'stats_db': os.getenv('STATS_DB', f'{pre}data/options/stats.db'),
            'vol_db': os.getenv('VOL_DB', f'{pre}data/options/vol.db'),
            'change_db': os.getenv('CHANGE_DB', f'{pre}data/options/option_change.db'),
            'option_db': os.getenv('OPTION_DB', f'{pre}data/options/options.db'),
            'dates_db': os.getenv('DATES_DB', f'{pre}data/options/dates.db'),
            
            # Earnings + Company Info
            'earnings_dict': os.getenv('EARNINGS_DICT', f'{pre}data/earnings/earnings.pkl'),
            'stock_names': os.getenv('STOCK_NAMES', f'{pre}data/stocks/stock_names.db'),
            'stock_info_dict': os.getenv('STOCK_INFO_DICT', f'{pre}data/stocks/stock_info.json'),
        }
        
        return connections
        
    except Exception as e:
        logger.error(f"Failed to get database paths: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Set up logging
        setup_logger()
        logger.info("\n12.8 Concentrate the mind upon Me, apply spiritual intelligence for Me; "
                   "verily you will reside with me after this existence without a doubt.\n")
        
        # Initialize the pipeline
        pipeline = Pipeline()
        
        # Run the pipeline
        # pipeline.master_run()
        
        # Print an option table
        df = pipeline.Optionsdb.get_option_chain('aapl')
        print(df)
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        sys.exit(1)
    finally:
        logging.shutdown()