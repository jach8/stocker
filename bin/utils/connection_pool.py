"""SQLite Connection Pool implementation with thread safety and monitoring"""

import os
import time
import sqlite3
import logging
import threading
import queue
from typing import Dict, Optional, Set
from dataclasses import dataclass
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from dotenv import load_dotenv

@dataclass
class ConnectionConfig:
    """Database connection configuration"""
    name: str
    path: str
    read_only: bool = False
    max_idle_time: int = 300  # 5 minutes
    max_lifetime: int = 3600  # 1 hour

class DatabaseConnection:
    """Wrapper for SQLite connection with metadata"""
    def __init__(self, conn: sqlite3.Connection, config: ConnectionConfig):
        self.connection = conn
        self.config = config
        self.created_at = time.time()
        self.last_used_at = time.time()
        self.in_use = False
        self.total_uses = 0

    def is_expired(self) -> bool:
        """Check if connection has exceeded its maximum lifetime"""
        return (time.time() - self.created_at) > self.config.max_lifetime

    def is_idle_expired(self) -> bool:
        """Check if connection has been idle for too long"""
        return (time.time() - self.last_used_at) > self.config.max_idle_time

class ConnectionPool:
    """Thread-safe connection pool for SQLite databases"""
    
    def __init__(self, min_connections: int = 1, max_connections: int = 10):
        self.logger = self._setup_logger()
        self.min_connections = min_connections
        self.max_connections = max_connections
        
        # Load database configurations from environment
        self._load_configurations()
        
        # Initialize pool state
        self.pools: Dict[str, queue.Queue] = {}
        self.active_connections: Dict[str, Set[DatabaseConnection]] = {}
        self.lock = threading.RLock()
        
        # Initialize connection pools
        self._initialize_pools()
        
        # Start maintenance thread
        self._start_maintenance_thread()
        
    def _setup_logger(self) -> logging.Logger:
        """Configure logging"""
        logger = logging.getLogger('ConnectionPool')
        logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        # File handler with rotation
        file_handler = RotatingFileHandler(
            'logs/connection_pool.log',
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

    def _load_configurations(self):
        """Load database configurations from environment variables"""
        load_dotenv()
        
        self.configs = {
            'daily': ConnectionConfig(
                'Daily Stocks',
                os.getenv('DAILY_DB', '')
            ),
            'intraday': ConnectionConfig(
                'Intraday Stocks',
                os.getenv('INTRADAY_DB', '')
            ),
            'options': ConnectionConfig(
                'Options',
                os.getenv('OPTION_DB', '')
            ),
            'options_ro': ConnectionConfig(
                'Options Read-Only',
                os.getenv('OPTION_DB', ''),
                read_only=True
            ),
            'changes': ConnectionConfig(
                'Options Changes',
                os.getenv('CHANGE_DB', '')
            ),
            'volume': ConnectionConfig(
                'Options Volume',
                os.getenv('VOL_DB', '')
            ),
            'stats': ConnectionConfig(
                'Options Stats',
                os.getenv('STATS_DB', '')
            ),
            'tracking': ConnectionConfig(
                'Tracking',
                os.getenv('TRACKING_DB', '')
            ),
            'tracking_values': ConnectionConfig(
                'Tracking Values',
                os.getenv('TRACKING_VALUES_DB', '')
            ), 
            'inactive': ConnectionConfig(
                'Inactive Options',
                os.getenv('INACTIVE_DB', '')
            ),
            'backup': ConnectionConfig(
                'Backup Options',
                os.getenv('BACKUP_DB', '')
            ),
            'dates': ConnectionConfig(
                'Dates',
                os.getenv('DATES_DB', '')
            ),
            'stock_names': ConnectionConfig(
                'Stock Names',
                os.getenv('STOCK_NAMES', '')
            ),
        }

    def _create_connection(self, config: ConnectionConfig) -> DatabaseConnection:
        """Create a new database connection"""
        try:
            if config.read_only:
                conn = sqlite3.connect(
                    f'file:{config.path}?mode=ro',
                    uri=True,
                    timeout=30
                )
            else:
                conn = sqlite3.connect(config.path, timeout=30)
            
            # Enable foreign keys and other pragmas
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            
            return DatabaseConnection(conn, config)
            
        except sqlite3.Error as e:
            self.logger.error(f"Failed to create connection for {config.name}: {str(e)}")
            raise

    def _initialize_pools(self):
        """Initialize connection pools with minimum connections"""
        for db_name, config in self.configs.items():
            if not config.path:
                continue
                
            self.pools[db_name] = queue.Queue()
            self.active_connections[db_name] = set()
            
            # Create minimum number of connections
            for _ in range(self.min_connections):
                try:
                    conn = self._create_connection(config)
                    self.pools[db_name].put(conn)
                except sqlite3.Error as e:
                    self.logger.error(f"Failed to initialize pool for {config.name}: {str(e)}")

    def _maintenance_task(self):
        """Periodic maintenance task to clean up idle and expired connections"""
        while True:
            try:
                with self.lock:
                    for db_name, pool in self.pools.items():
                        # Check idle connections in pool
                        idle_connections = []
                        while not pool.empty():
                            conn = pool.get_nowait()
                            if conn.is_idle_expired() or conn.is_expired():
                                conn.connection.close()
                            else:
                                idle_connections.append(conn)
                        
                        # Put back non-expired connections
                        for conn in idle_connections:
                            pool.put(conn)
                            
                        # Check active connections
                        expired = {conn for conn in self.active_connections[db_name] 
                                if conn.is_expired()}
                        for conn in expired:
                            self.logger.warning(
                                f"Force closing expired active connection for {conn.config.name}"
                            )
                            conn.connection.close()
                            self.active_connections[db_name].remove(conn)
                
                # Run maintenance every minute
                time.sleep(60)
                
            except Exception as e:
                self.logger.error(f"Error in maintenance task: {str(e)}")
                time.sleep(60)  # Wait before retrying

    def _start_maintenance_thread(self):
        """Start the maintenance thread"""
        maintenance_thread = threading.Thread(
            target=self._maintenance_task,
            daemon=True,
            name="ConnectionPoolMaintenance"
        )
        maintenance_thread.start()

    @contextmanager
    def get_connection(self, db_name: str) -> sqlite3.Connection:
        """
        Get a connection from the pool
        
        Args:
            db_name: Name of the database to connect to
            
        Yields:
            SQLite connection
            
        Raises:
            KeyError: If database name is invalid
            sqlite3.Error: If connection fails
        """
        if db_name not in self.configs:
            raise KeyError(f"Invalid database name: {db_name}")
            
        conn = None
        try:
            with self.lock:
                # Try to get connection from pool
                try:
                    conn = self.pools[db_name].get_nowait()
                    
                    # Check if connection is still valid
                    if conn.is_expired():
                        conn.connection.close()
                        conn = self._create_connection(self.configs[db_name])
                        
                except queue.Empty:
                    # Create new connection if under max limit
                    total_connections = (len(self.active_connections[db_name]) + 
                                      self.pools[db_name].qsize())
                    
                    if total_connections < self.max_connections:
                        conn = self._create_connection(self.configs[db_name])
                    else:
                        # Wait for an available connection
                        conn = self.pools[db_name].get(timeout=30)
                
                conn.in_use = True
                conn.last_used_at = time.time()
                conn.total_uses += 1
                self.active_connections[db_name].add(conn)
            
            try:
                yield conn.connection
            finally:
                with self.lock:
                    if conn in self.active_connections[db_name]:
                        conn.in_use = False
                        self.active_connections[db_name].remove(conn)
                        self.pools[db_name].put(conn)
                    
        except Exception as e:
            self.logger.error(f"Error managing connection for {db_name}: {str(e)}")
            if conn:
                try:
                    conn.connection.close()
                except:
                    pass
            raise

    def get_pool_status(self) -> Dict[str, Dict]:
        """Get current status of all connection pools"""
        status = {}
        with self.lock:
            for db_name, pool in self.pools.items():
                active = self.active_connections[db_name]
                status[db_name] = {
                    'available': pool.qsize(),
                    'in_use': len(active),
                    'total': pool.qsize() + len(active),
                    'max_connections': self.max_connections,
                    'oldest_connection': min(
                        (c.created_at for c in active | 
                         {p for p in (pool.queue if hasattr(pool, 'queue') else [])})
                        , default=None
                    )
                }
        return status

    def close_all(self):
        """Close all connections in all pools"""
        with self.lock:
            for db_name, pool in self.pools.items():
                # Close active connections
                for conn in self.active_connections[db_name]:
                    try:
                        conn.connection.close()
                    except:
                        pass
                self.active_connections[db_name].clear()
                
                # Close pooled connections
                while not pool.empty():
                    try:
                        conn = pool.get_nowait()
                        conn.connection.close()
                    except:
                        pass
                        
        self.logger.info("All connections closed")

# Global connection pool instance
_pool: Optional[ConnectionPool] = None

def get_pool() -> ConnectionPool:
    """Get or create the global connection pool instance"""
    global _pool
    if _pool is None:
        _pool = ConnectionPool()
    return _pool

if __name__ == "__main__":
    # Example usage
    pool = get_pool()
    
    try:
        # Get connection from pool
        with pool.get_connection('daily') as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT 1')
            result = cursor.fetchone()
            print(f"Test query result: {result}")
            
        # Print pool status
        status = pool.get_pool_status()
        print("\nPool Status:")
        for db_name, stats in status.items():
            print(f"\n{db_name}:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
                
    finally:
        pool.close_all()