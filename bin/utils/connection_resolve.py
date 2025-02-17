import os
from typing import Dict
from dotenv import load_dotenv
from bin.utils.Initialize import Initialize



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