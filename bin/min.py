import pickle
import pandas as pd 
import sqlite3 
import os 
import json 
import datetime as dt 
import numpy as np 
import sys
import logging
from typing import Dict, Optional
from pathlib import Path
from dotenv import load_dotenv

from price.db_connect import Prices
from options.manage_all import Manager as Optionsdb
from earnings.get_earnings import Earnings 
from price.perf.report import perf as performance
from alerts.options_alerts import Notifications
from signals.plays.ling import Scanner
from signals.plays.dxp import dxp 
from utils.Initialize import Initialize
from utils.add_stocks import add_stock
from utils.connection_pool import get_pool, ConnectionPool

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s: %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def load_environment() -> bool:
    """
    Load environment variables from .env file.
    
    Returns:
        bool: True if environment loaded successfully, False otherwise
    """
    try:
        # Load .env file
        if not load_dotenv():
            logger.error("Failed to load .env file")
            return False
            
        # Verify required environment variables
        required_vars = [
            'DAILY_DB',
            'INTRADAY_DB',
            'OPTION_DB',
            'CHANGE_DB',
            'VOL_DB',
            'STATS_DB',
            'TRACKING_DB',
            'TRACKING_VALUES_DB'
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error loading environment: {str(e)}")
        return False

def is_initialized() -> bool:
    """
    Check if the system has already been initialized.
    
    Returns:
        bool: True if system is initialized, False otherwise
    """
    try:
        # Check for key database files
        paths_to_check = [
            os.getenv('DAILY_DB'),
            os.getenv('OPTION_DB'),
            os.getenv('STOCK_NAMES', 'data/stocks/stock_names.db')
        ]
        
        return all(os.path.exists(path) for path in paths_to_check if path)
        
    except Exception as e:
        logger.error(f"Error checking initialization state: {str(e)}")
        return False

def init() -> bool:
    """
    Initialize the program if not already initialized.
    
    Returns:
        bool: True if initialization successful or already initialized, False otherwise
    """
    try:
        if is_initialized():
            logger.info("System already initialized")
            return True
            
        logger.info("Initializing system...")
        Initialize()
        return True
        
    except Exception as e:
        logger.error(f"Initialization failed: {str(e)}")
        return False

def get_path(pre: str = '') -> Dict[str, str]:
    """
    Get database paths dictionary from environment or defaults.
    
    Args:
        pre: Optional prefix for paths
        
    Returns:
        Dictionary of connection paths
    
    Raises:
        ValueError: If required paths are missing
    """
    try:
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
        
        # Validate required paths
        missing_paths = [k for k, v in connections.items() if not v]
        if missing_paths:
            raise ValueError(f"Missing required paths: {', '.join(missing_paths)}")
            
        return connections
        
    except Exception as e:
        logger.error(f"Error getting paths: {str(e)}")
        raise

def check_path(connections: Dict[str, str]) -> bool:
    """
    Check if all files in connections dictionary exist.
    
    Args:
        connections: Dictionary of connection paths
        
    Returns:
        True if all files exist
        
    Raises:
        ValueError: If any required file is missing
    """
    try:
        missing_files = []
        for key, value in connections.items():
            if not os.path.exists(value):
                missing_files.append(value)
                
        if missing_files:
            raise ValueError(f"Missing files: {', '.join(missing_files)}")
            
        return True
        
    except Exception as e:
        logger.error(f"Path check failed: {str(e)}")
        raise

class Manager:
    """
    Manager class for handling all database connections and operations.
    Uses connection pooling for efficient database access.
    """
    
    def __init__(self, connections: Optional[Dict[str, str]] = None):
        """
        Initialize the Manager with connection pooling.
        
        Args:
            connections: Optional dictionary of connection paths.
                       If string is provided, it's used as a prefix.
                       If None, default paths are used.
                       
        Raises:
            ValueError: If connections are invalid or initialization fails
        """
        try:
            # Handle different input types for connections
            if isinstance(connections, str):
                connections = get_path(connections)
            elif connections is None:
                connections = get_path()
            elif not isinstance(connections, dict):
                raise ValueError('Connections must be a dictionary')
                
            # Check if the files exist
            check_path(connections)
            
            # Initialize the connection pool
            self.pool = get_pool()
            
            # Store paths for components that might need direct access
            self.connection_paths = connections
            
            # Initialize components with connection paths
            # They will use the connection pool internally
            self.Optionsdb = Optionsdb(connections)
            self.Pricedb = Prices(connections)
            self.performance = performance(connections)
            self.Notifications = Notifications(connections)
            self.Scanner = Scanner(connections)
            self.dxp = dxp(connections)
            
            logger.info("Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Manager initialization failed: {str(e)}")
            self.close_connection()  # Cleanup on failure
            raise
    
    def close_connection(self):
        """Close all database connections in the pool."""
        try:
            # Close connections in components
            if hasattr(self, 'Optionsdb'):
                self.Optionsdb.close_connections()
            if hasattr(self, 'Pricedb'):
                self.Pricedb.close_connections()
            
            # Close all pooled connections
            if hasattr(self, 'pool'):
                self.pool.close_all()
                
        except Exception as e:
            logger.error(f"Error closing connections: {str(e)}")
            raise
    
    def addStock(self, stock: str):
        """
        Add a new stock to the database.
        
        Args:
            stock: Stock symbol to add
            
        Raises:
            Exception: If stock addition fails
        """
        try:
            # Use connection pool for stock_names database
            with self.pool.get_connection('stock_names') as conn:
                add_stock(conn, self.connection_paths['ticker_path'], stock)
                logger.info(f"Successfully added stock: {stock}")
                
        except Exception as e:
            logger.error(f"Failed to add stock {stock}: {str(e)}")
            raise

def main():
    """Main entry point for the application."""
    manager = None
    try:
        # Load environment variables
        if not load_environment():
            sys.exit(1)
        
        # Initialize if needed
        if not init():
            sys.exit(1)
        
        # Create manager instance
        manager = Manager()
        logger.info("System ready")
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)
        
    finally:
        # Ensure proper cleanup
        if manager:
            try:
                manager.close_connection()
            except Exception as e:
                logger.error(f"Error during cleanup: {str(e)}")
                sys.exit(1)

if __name__ == "__main__":
    main()