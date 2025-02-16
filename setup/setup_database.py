#!/usr/bin/env python3
"""Database setup and validation script"""

import os
import sys
import json
import sqlite3
import logging
import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass
from dotenv import load_dotenv

@dataclass
class DatabaseConfig:
    """Database configuration class"""
    name: str
    path: str
    schema: Dict[str, List[str]]
    required: bool = True
    read_only: bool = False

class DatabaseSetup:
    """Handles database setup, validation, and migrations"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.load_env()
        self._init_db_configs()
        
    def _setup_logger(self) -> logging.Logger:
        """Configure logging"""
        logger = logging.getLogger('DatabaseSetup')
        logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        # File handler with rotation
        file_handler = RotatingFileHandler(
            'logs/setup.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger

    def load_env(self):
        """Load environment variables"""
        env_path = Path('.env')
        if not env_path.exists():
            self.logger.error('.env file not found')
            raise FileNotFoundError('.env file not found')
            
        load_dotenv()
        self.logger.info('Loaded environment variables')

    def _init_db_configs(self):
        """Initialize database configurations"""
        # Schema definitions
        stocks_schema = {
            'stocks': [
                'date TEXT PRIMARY KEY',
                'open REAL',
                'high REAL',
                'low REAL',
                'close REAL',
                'volume INTEGER'
            ]
        }
        
        options_schema = {
            'options': [
                'contractSymbol TEXT PRIMARY KEY',
                'strike REAL',
                'expiry TEXT',
                'lastPrice REAL',
                'bid REAL',
                'ask REAL',
                'volume INTEGER',
                'openInterest INTEGER',
                'impliedVolatility REAL'
            ]
        }

        tracking_schema = {
            'tracking': [
                'id INTEGER PRIMARY KEY AUTOINCREMENT',
                'symbol TEXT',
                'date TEXT',
                'value REAL'
            ]
        }
        
        # Initialize database configurations
        self.db_configs = [
            DatabaseConfig('Daily Stocks', os.getenv('DAILY_DB', ''), stocks_schema),
            DatabaseConfig('Intraday Stocks', os.getenv('INTRADAY_DB', ''), stocks_schema),
            DatabaseConfig('Options', os.getenv('OPTION_DB', ''), options_schema),
            DatabaseConfig('Options Changes', os.getenv('CHANGE_DB', ''), options_schema),
            DatabaseConfig('Options Volume', os.getenv('VOL_DB', ''), {'volume': ['date TEXT', 'value INTEGER']}),
            DatabaseConfig('Options Stats', os.getenv('STATS_DB', ''), {'stats': ['date TEXT', 'value REAL']}),
            DatabaseConfig('Tracking', os.getenv('TRACKING_DB', ''), tracking_schema),
            DatabaseConfig('Tracking Values', os.getenv('TRACKING_VALUES_DB', ''), tracking_schema),
            DatabaseConfig('Backup', os.getenv('BACKUP_DB', ''), {}, required=False),
            DatabaseConfig('Inactive', os.getenv('INACTIVE_DB', ''), {}, required=False)
        ]

    def ensure_directories(self):
        """Create necessary directories if they don't exist"""
        required_dirs = [
            'data/prices',
            'data/options',
            'data/options/log',
            'data/stocks',
            'data/bonds',
            'data/earnings',
            'logs'
        ]
        
        for dir_path in required_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            self.logger.info(f'Ensured directory exists: {dir_path}')

    def validate_connection_string(self, db_config: DatabaseConfig) -> bool:
        """Validate database connection string"""
        if not db_config.path:
            if db_config.required:
                self.logger.error(f'Missing connection string for {db_config.name}')
                return False
            return True
            
        path = Path(db_config.path)
        if not path.parent.exists():
            self.logger.error(f'Parent directory does not exist for {db_config.name}: {path.parent}')
            return False
            
        return True

    def test_connection(self, db_config: DatabaseConfig) -> bool:
        """Test database connection"""
        if not db_config.path or not db_config.required:
            return True
            
        try:
            conn = sqlite3.connect(db_config.path)
            cursor = conn.cursor()
            cursor.execute('SELECT 1')
            conn.close()
            self.logger.info(f'Successfully connected to {db_config.name}')
            return True
        except sqlite3.Error as e:
            self.logger.error(f'Failed to connect to {db_config.name}: {str(e)}')
            return False

    def create_database(self, db_config: DatabaseConfig) -> bool:
        """Create database and its schema if it doesn't exist"""
        if not db_config.path or not db_config.required:
            return True
            
        try:
            conn = sqlite3.connect(db_config.path)
            cursor = conn.cursor()
            
            # Create tables
            for table_name, columns in db_config.schema.items():
                create_table_sql = f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    {', '.join(columns)}
                )
                """
                cursor.execute(create_table_sql)
            
            conn.commit()
            conn.close()
            self.logger.info(f'Successfully created/validated schema for {db_config.name}')
            return True
        except sqlite3.Error as e:
            self.logger.error(f'Failed to create/validate schema for {db_config.name}: {str(e)}')
            return False

    def verify_permissions(self, db_config: DatabaseConfig) -> bool:
        """Verify database permissions"""
        if not db_config.path or not db_config.required:
            return True
            
        try:
            # Test write access
            conn = sqlite3.connect(db_config.path)
            cursor = conn.cursor()
            
            # Try to create and drop a test table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS _test_permissions (
                id INTEGER PRIMARY KEY
            )
            """)
            cursor.execute("DROP TABLE _test_permissions")
            
            conn.commit()
            conn.close()
            
            # Test read-only access if specified
            if db_config.read_only:
                conn = sqlite3.connect(f'file:{db_config.path}?mode=ro', uri=True)
                cursor = conn.cursor()
                cursor.execute('SELECT 1')
                conn.close()
                
            self.logger.info(f'Successfully verified permissions for {db_config.name}')
            return True
        except sqlite3.Error as e:
            self.logger.error(f'Failed to verify permissions for {db_config.name}: {str(e)}')
            return False

    def validate_json_files(self) -> bool:
        """Validate JSON configuration files"""
        json_files = {
            'Tickers': os.getenv('TICKER_PATH', ''),
            'Stock Info': os.getenv('STOCK_INFO_DICT', '')
        }
        
        success = True
        for name, path in json_files.items():
            if not path:
                self.logger.error(f'Missing path for {name} JSON file')
                success = False
                continue
                
            try:
                with open(path, 'r') as f:
                    json.load(f)
                self.logger.info(f'Successfully validated {name} JSON file')
            except (FileNotFoundError, json.JSONDecodeError) as e:
                self.logger.error(f'Failed to validate {name} JSON file: {str(e)}')
                success = False
                
        return success

    def setup(self) -> Tuple[bool, Dict[str, str]]:
        """Run complete database setup"""
        start_time = dt.datetime.now()
        self.logger.info(f'Starting database setup at {start_time}')
        
        status = {}
        success = True
        
        # Create required directories
        try:
            self.ensure_directories()
            status['directories'] = 'Created successfully'
        except Exception as e:
            self.logger.error(f'Failed to create directories: {str(e)}')
            status['directories'] = f'Failed: {str(e)}'
            success = False

        # Validate JSON files
        json_valid = self.validate_json_files()
        status['json_files'] = 'Validated successfully' if json_valid else 'Validation failed'
        success = success and json_valid

        # Process each database
        for db_config in self.db_configs:
            db_success = True
            
            # Validate connection string
            if not self.validate_connection_string(db_config):
                db_success = False
            
            # Create/validate schema
            if db_success and not self.create_database(db_config):
                db_success = False
                
            # Test connection
            if db_success and not self.test_connection(db_config):
                db_success = False
                
            # Verify permissions
            if db_success and not self.verify_permissions(db_config):
                db_success = False
                
            status[db_config.name] = 'Setup successful' if db_success else 'Setup failed'
            success = success and db_success

        end_time = dt.datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        status['duration'] = f'{duration:.2f} seconds'
        status['overall'] = 'Success' if success else 'Failed'
        
        self.logger.info(f'Database setup completed in {duration:.2f} seconds')
        self.logger.info(f'Overall status: {status["overall"]}')
        
        return success, status

def main():
    """Main entry point"""
    setup = DatabaseSetup()
    success, status = setup.setup()
    
    # Print summary
    print("\nSetup Summary:")
    print("=" * 50)
    for key, value in status.items():
        print(f"{key:.<30} {value}")
    print("=" * 50)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()