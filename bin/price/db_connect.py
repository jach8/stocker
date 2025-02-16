from typing import Dict, List, Optional, Union, Any, Tuple, ContextManager
import numpy as np
import pandas as pd
import sqlite3 as sql
import datetime as dt
from tqdm import tqdm
import time
import json
import logging
from logging.handlers import RotatingFileHandler
import sys
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from bin.price.indicators import Indicators
from bin.price.get_data import UpdateStocks
from bin.utils.connection_pool import get_pool

# Custom exceptions
class DatabaseConnectionError(Exception):
    """Raised when database connection fails"""
    pass

class QueryExecutionError(Exception):
    """Raised when SQL query execution fails"""
    pass

class InvalidParameterError(Exception):
    """Raised when invalid parameters are provided"""
    pass

# Configure logging
def setup_logger(name: str) -> logging.Logger:
    """Configure logger with file and console handlers"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # File handler with rotation
    # log_file = Path("logs/price_db.log")
    # log_file.parent.mkdir(exist_ok=True)
    # file_handler = RotatingFileHandler(
    #     log_file,
    #     maxBytes=10*1024*1024,  # 10MB
    #     backupCount=5
    # )
    # file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    # file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    
    # logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logger(__name__)

class Prices(UpdateStocks):
    """Class for managing stock price database connections and queries"""
    
    def __init__(self, connections: Dict[str, str]) -> None:
        """
        Initialize price database manager
        
        Args:
            connections: Dictionary containing database connection paths
        
        Raises:
            DatabaseConnectionError: If database connection fails
            FileNotFoundError: If required files are not found
        """
        super().__init__(connections)
        self.execution_start_time = time.time()
        self.pool = get_pool()
        
        try:
            # Validate connection parameters
            required_keys = ['stock_names', 'daily_db', 'intraday_db', 'ticker_path']
            if not all(key in connections for key in required_keys):
                raise InvalidParameterError(f"Missing required connection parameters: {required_keys}")
            
            # Validate file existence
            for key, path in connections.items():
                if not Path(path).exists():
                    raise FileNotFoundError(f"File not found: {path} for {key}")
            
            # Store database mapping
            self.db_mapping = {
                'stock_names': 'stock_names',
                'daily': 'daily',
                'intraday': 'intraday'
            }
            
            # Test all database connections
            for db_type in self.db_mapping.keys():
                with self._get_connection(db_type) as conn:
                    conn.execute("SELECT 1")
            
            # Load ticker data
            try:
                with open(connections['ticker_path'], 'r') as f:
                    self.stocks = json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse ticker JSON file: {e}")
                raise
                
            self.Indicators = Indicators
            
            logger.info(f"PriceDB Initialized successfully at {dt.datetime.now()}")
            logger.info("Connection pool initialized")
            
        except (sql.Error, FileNotFoundError, json.JSONDecodeError) as e:
            error_msg = f"Initialization failed: {str(e)}"
            logger.error(error_msg)
            raise DatabaseConnectionError(error_msg) from e
    
    @contextmanager
    def _get_connection(self, db_type: str) -> ContextManager[sql.Connection]:
        """
        Get a database connection from the pool
        
        Args:
            db_type: Type of database ('names', 'daily', or 'intraday')
            
        Returns:
            SQLite connection object from pool
            
        Raises:
            DatabaseConnectionError: If connection fails
        """
        if db_type not in self.db_mapping:
            raise InvalidParameterError(f"Invalid database type: {db_type}")
            
        try:
            with self.pool.get_connection(self.db_mapping[db_type]) as conn:
                yield conn
        except (sql.Error, KeyError) as e:
            error_msg = f"Failed to get {db_type} database connection: {str(e)}"
            logger.error(error_msg)
            raise DatabaseConnectionError(error_msg) from e
            
    def update_stock_prices(self) -> None:
        """Update stock prices in database"""
        logger.info('Starting stock price update')
        try:
            self.update()
            logger.info('Successfully updated stock prices')
        except Exception as e:
            logger.error(f'Failed to update stock prices: {str(e)}')
            raise
            
    def custom_q(self, q: str) -> pd.DataFrame:
        """
        Execute a custom query on the daily_db
        
        Args:
            q: SQL query string
            
        Returns:
            DataFrame containing query results
            
        Raises:
            QueryExecutionError: If query execution fails
        """
        try:
            with self._get_connection('daily') as conn:
                cursor = conn.cursor()
                cursor.execute(q)
                results = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                return pd.DataFrame(results, columns=columns)
        except sql.Error as e:
            error_msg = f"Query execution failed: {str(e)}\nQuery: {q}"
            logger.error(error_msg)
            raise QueryExecutionError(error_msg) from e
    
    def _get1minCl(self, stock: str, agg: str = '1min') -> pd.DataFrame:
        """Get 1-minute close prices for a stock"""
        try:
            q = f'''select datetime(date) as date, close from {stock} order by datetime(date) asc'''
            with self._get_connection('intraday') as conn:
                cursor = conn.cursor()
                cursor.execute(q)
                df = pd.DataFrame(cursor.fetchall(), columns=['date', stock])
                df.date = pd.to_datetime(df.date)
                df = df.set_index('date')
                if agg != '1min':
                    df = df.resample(agg).last()
                return df
        except (sql.Error, pd.errors.EmptyDataError) as e:
            logger.error(f"Failed to get 1-minute close for {stock}: {str(e)}")
            raise

    def get_intraday_close(self, stocks: List[str], agg: str = '1min') -> pd.DataFrame:
        """
        Get intraday closing prices for multiple stocks
        
        Args:
            stocks: List of stock symbols
            agg: Aggregation interval
            
        Returns:
            DataFrame with stock prices
            
        Raises:
            InvalidParameterError: If stocks is not a list
        """
        if not isinstance(stocks, list):
            raise InvalidParameterError("Input must be a list of stocks")
            
        try:
            out = [self._get1minCl(stock) for stock in stocks]
            out = [i.resample(agg).last() for i in out]
            return pd.concat(out, axis=1)
        except Exception as e:
            logger.error(f"Failed to get intraday close prices: {str(e)}")
            raise

    def _getClose(self, stock: str) -> pd.DataFrame:
        """Get daily closing prices for a stock"""
        try:
            q = f'''select date(date) as date, close as "Close" from {stock} order by date(date) asc'''
            with self._get_connection('daily') as conn:
                df = pd.read_sql_query(q, conn, parse_dates=['date'], index_col='date')
                return df.rename(columns={'Close': stock})
        except (sql.Error, pd.errors.DatabaseError) as e:
            logger.error(f"Failed to get close prices for {stock}: {str(e)}")
            raise

    def get_close(self, stocks: List[str], start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        """Get daily closing prices for multiple stocks"""
        if not isinstance(stocks, list):
            raise InvalidParameterError("Input must be a list of stocks")
            
        try:
            out = [self._getClose(stock) for stock in stocks]
            df = pd.concat(out, axis=1)
            
            if start is not None:
                df = df[df.index >= start]
            if end is not None:
                df = df[df.index <= end]
                
            return df
        except Exception as e:
            logger.error(f"Failed to get close prices: {str(e)}")
            raise

    def ohlc(self, stock: str, daily: bool = True, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        """Get OHLCV data for a stock"""
        try:
            if daily:
                q = f'''select date(date) as "Date", open, high, low, close, volume from {stock} order by date(date) asc'''
                db_type = 'daily'
            else:
                if start is None:
                    q = f'''select datetime(date) as "Date", open, high, low, close, volume from {stock} order by datetime(date) asc'''
                else:
                    q = f'''
                    select
                        datetime(date) as "Date",
                        open,
                        high,
                        low,
                        close,
                        volume
                    from {stock}
                    where
                        date(date) >= date("{start}")
                    order by datetime(date) asc'''
                db_type = 'intraday'
                
            with self._get_connection(db_type) as conn:
                cursor = conn.cursor()
                cursor.execute(q)
                df = pd.DataFrame(cursor.fetchall(), columns=[i[0] for i in cursor.description])
                df.Date = pd.to_datetime(df.Date)
                df.index = df.Date
                
                if start is not None:
                    df = df[df.index >= start]
                if end is not None:
                    df = df[df.index <= end]
                    
                return df
                
        except Exception as e:
            logger.error(f"Failed to get OHLC data for {stock}: {str(e)}")
            raise

    def get_aggregates(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Get daily, weekly, monthly aggregates
        
        Args:
            df: DataFrame with datetime index
            
        Returns:
            Dictionary of aggregated DataFrames
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise InvalidParameterError("DataFrame index must be DatetimeIndex")
            
        try:
            daily = df.resample('B').last().dropna()
            weekly = df.resample('W').last().dropna()
            monthly = df.resample('M').last().dropna()
            return {'B': daily, 'W': weekly, 'M': monthly}
        except Exception as e:
            logger.error(f"Failed to compute aggregates: {str(e)}")
            raise

    def intra_day_aggs(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Get intraday aggregations"""
        df.index = pd.to_datetime(df.index)
        if not isinstance(df.index, pd.DatetimeIndex):
            raise InvalidParameterError("DataFrame index must be DatetimeIndex")
            
        try:
            return {
                '3min': df.resample('3T').last().dropna(),
                '6min': df.resample('6T').last().dropna(),
                '18min': df.resample('18T').last().dropna(),
                '1H': df.resample('H').last().dropna(),
                '4H': df.resample('4H').last().dropna()
            }
        except Exception as e:
            logger.error(f"Failed to compute intraday aggregates: {str(e)}")
            raise

    def daily_aggregates(self, stock: str) -> Dict[str, pd.DataFrame]:
        """Get daily aggregates for a stock"""
        try:
            df = self._getClose(stock)
            return self.get_aggregates(df)
        except Exception as e:
            logger.error(f"Failed to get daily aggregates for {stock}: {str(e)}")
            raise

    def get_indicators(self, 
                      stock: str, 
                      daily: bool = True, 
                      kwargs: Optional[Dict[str, int]] = None, 
                      start: Optional[str] = None, 
                      end: Optional[str] = None, 
                      close_only: bool = True, 
                      resample_timeframe: Optional[str] = None) -> pd.DataFrame:
        """Get technical indicators for a stock"""
        try:
            if kwargs is None:
                kwargs = dict(fast=2, medium=3, slow=5, m=2)
                
            if not daily:
                daily_df = self.ohlc(stock, True,)    
                G = Indicators(daily_df)
                daily_smas = G._get_moving_averages(fast=kwargs['fast'], medium=kwargs['medium'], slow=kwargs['slow'])
                dsma = pd.DataFrame(daily_smas, index=daily_df.index, columns=list(daily_smas.keys()))
                dsma.index = pd.to_datetime(dsma.index)
                
                colmaps = {
                    '_fast': f"{kwargs['fast']}D",
                    '_med': f"{kwargs['medium']}D",
                    '_slow': f"{kwargs['slow']}D"
                }
                
                fast_cols = dsma.columns.str.contains('_fast')
                medium_cols = dsma.columns.str.contains('_med')
                slow_cols = dsma.columns.str.contains('_slow')
                
                fc = {x: x.replace('_fast', colmaps['_fast']) for x in dsma.columns[fast_cols]}
                mc = {x: x.replace('_med', colmaps['_med']) for x in dsma.columns[medium_cols]}
                sc = {x: x.replace('_slow', colmaps['_slow']) for x in dsma.columns[slow_cols]}
                dsma.rename(columns={**fc, **mc, **sc}, inplace=True)
                self.daily_smas = dsma
                
                if close_only:
                    time_frame = resample_timeframe or '1min'
                    df = self._get1minCl(stock, agg=time_frame)[stock]
                    i = Indicators(df)
                    out = i._get_moving_averages()
                    out = pd.DataFrame(out, index=df.index, columns=list(out.keys()))
                else:
                    df = self.ohlc(stock, daily, start, end)
                    i = Indicators(df)
                    self.Indicators = i
                    out = i.indicator_df(fast=kwargs['fast'], medium=kwargs['medium'], slow=kwargs['slow'], m=kwargs['m'])
                    
                dsma['date_day'] = dsma.index.date
                out['date_day'] = out.index.date
                out['Date'] = out.index
                dsma.rename(columns={'close': 'close_daily'}, inplace=True)
                out.rename(columns={'close': 'close'}, inplace=True)
                out = pd.merge(out, dsma, on='date_day', how='left').drop(columns=['date_day'])
                return out.set_index('Date')
            else:    
                df = self.ohlc(stock, daily, start, end)
                i = Indicators(df)
                self.Indicators = i
                return i.indicator_df(fast=kwargs['fast'], medium=kwargs['medium'], slow=kwargs['slow'], m=kwargs['m'])
        
        except Exception as e:
            logger.error(f"Failed to get indicators for {stock}: {str(e)}")
            raise

    def close_connections(self) -> None:
        """Close all pooled connections"""
        try:
            self.pool.close_all()
            
            end_time = time.time()
            runtime_min = (end_time - self.execution_start_time) / 60
            logger.info(f"All connections returned to pool. Total runtime: {runtime_min:.2f} min")
            
        except Exception as e:
            logger.error(f"Error closing pool connections: {str(e)}")
            raise

if __name__ == "__main__":
    print("\n(26) To whatever and wherever the restless and unsteady mind wanders this mind should be restrained then and there and brought under the control of the self alone. (And nothing else) \n")
    
    connections = {
        'daily_db': 'data/prices/stocks.db',
        'intraday_db': 'data/prices/stocks_intraday.db',
        'ticker_path': 'data/stocks/tickers.json',
        'stock_names': 'data/stocks/stock_names.db'
    }
    
    try:
        m = Prices(connections)
        print('\n\n\n')
        result = m.get_indicators('spy', daily=False, start="2025-01-20")
        print(result)
        m.close_connections()
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        raise