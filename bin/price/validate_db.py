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
from dataclasses import dataclass

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
        
        # Console handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger

    def load_env(self):
        """Load database connection configuration"""
        config_path = Path('connections.json')
        if not config_path.exists():
            self.logger.error('connections.json file not found')
            raise FileNotFoundError('connections.json file not found')
            
        self.logger.info('Database configuration file found')

    def _init_db_configs(self):
        """Initialize database configurations"""
        # Load database configurations from connections.json
        with open('connections.json', 'r') as f:
            connections = json.load(f)

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
        
        # Initialize database configurations
        self.db_configs = [
            DatabaseConfig('Daily Stocks', connections.get('daily_db', ''), stocks_schema),
            DatabaseConfig('Intraday Stocks', connections.get('intraday_db', ''), stocks_schema),
            DatabaseConfig('Stock Names', connections.get('stock_names', ''), {'stocks': ['symbol TEXT PRIMARY KEY', 'name TEXT']})
        ]

    def ensure_directories(self):
        """Create necessary directories if they don't exist"""
        required_dirs = [
            'data/prices',
            'data/stocks',
            'data/earnings'
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
        try:
            with open('connections.json', 'r') as f:
                connections = json.load(f)
                
            json_files = {
                'Tickers': connections.get('ticker_path', ''),
                'Stock Info': connections.get('stock_info_dict', '')
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
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.error(f'Failed to load connections.json: {str(e)}')
            return False

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