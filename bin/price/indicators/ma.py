"""Moving Averages Signals.

This module implements technical analysis tools for calculating and analyzing
moving averages on financial time series data.

Key Features:
- Multiple moving average types (SMA, EMA, WMA, KAMA)
- Automatic timeframe detection
- Moving average ribbons
- Signal generation based on MA crossovers and convergence

Example:
    ma = moving_avg()
    df = get_price_data()  # Get OHLCV data
    ribbon = ma.ribbon(df, ma='ema')  # Generate MA ribbon
    signals = ma.generate_signals(ribbon)  # Generate trading signals
"""

from typing import Union, Optional, Dict, List
import pandas as pd 
import numpy as np 
import sqlite3 as sql 
from logging import getLogger

logger = getLogger(__name__)

class MovingAverageError(Exception):
    """Base exception for moving average calculation errors."""
    pass

class TimeframeError(MovingAverageError):
    """Exception raised for timeframe detection errors."""
    pass

class DataValidationError(MovingAverageError):
    """Exception raised for invalid input data."""
    pass

class moving_avg:
    """Calculate and analyze various types of moving averages.

    This class implements different moving average types and provides tools
    for technical analysis using moving averages.

    Attributes:
        windows (np.ndarray): Array of periods for MA calculations
            Default periods: [6, 10, 20, 28, 96, 108, 200, 496]

    Methods:
        sma: Simple Moving Average
        ema: Exponential Moving Average
        wma: Weighted Moving Average
        kama: Kaufman Adaptive Moving Average
        ribbon: Generate multiple MAs as a ribbon
        generate_signals: Create trading signals from MAs
    """

    def __init__(self) -> None:
        """Initialize moving average calculator with default periods."""
        self.windows: np.ndarray = np.array([6, 10, 20, 28, 50, 96, 108, 200, 496])

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """Validate input DataFrame structure and content.

        Args:
            df: Input DataFrame to validate

        Raises:
            DataValidationError: If DataFrame doesn't meet requirements
        """
        if not isinstance(df, pd.DataFrame):
            raise DataValidationError("Input must be a pandas DataFrame")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise DataValidationError("DataFrame must have DatetimeIndex")
        if len(df) < 2:
            raise DataValidationError("DataFrame must have at least 2 rows")
        if 'close' not in list(df.columns.str.lower()):
            raise DataValidationError("DataFrame must have 'close' column")

    def get_time_difference(self, df: pd.DataFrame) -> str:
        """Determine time difference between consecutive rows.

        Args:
            df: DataFrame with datetime index

        Returns:
            Single character timeframe indicator:
                'T': Minutes
                'H': Hours
                'D': Days
                'W': Weeks
                'M': Months

        Raises:
            TimeframeError: If time difference cannot be determined
        """
        try:
            diff = df.index[-1] - df.index[-2]
            if diff.days >= 28:
                return 'M'
            elif diff.days >= 7:
                return 'W'
            elif diff.days >= 1:
                return 'D'
            elif diff.seconds >= 3600:
                return 'H'
            else:
                return 'T'
        except Exception as e:
            logger.error(f"Error calculating time difference: {str(e)}")
            raise TimeframeError(f"Failed to determine timeframe: {str(e)}")

    def derive_timeframe(self, df: pd.DataFrame) -> str:
        """Get DataFrame frequency indicator.

        Args:
            df: DataFrame with datetime index

        Returns:
            Single character timeframe indicator

        Raises:
            TimeframeError: If frequency cannot be determined
        """
        try:
            freq = pd.infer_freq(df.index)
            if freq is None:
                freq = self.get_time_difference(df)
            return freq
        except Exception as e:
            logger.error(f"Error deriving timeframe: {str(e)}")
            raise TimeframeError(f"Failed to derive timeframe: {str(e)}")

    def ema(self, df: pd.DataFrame, window: int) -> pd.Series:
        """Calculate Exponential Moving Average.

        Uses Wilder's smoothing method (1/window decay factor).

        Args:
            df: DataFrame with price data
            window: Moving average period

        Returns:
            Series containing EMA values with name format 'EMA{window}{timeframe}'
        """
        self._validate_dataframe(df)
        out = df.copy()
        tf = self.derive_timeframe(df)
        col_name = f'EMA{window}{tf}'
        out[col_name] = df['close'].ewm(span=window, adjust=False).mean()
        return out[col_name]

    def sma(self, df: pd.DataFrame, window: int) -> pd.Series:
        """Calculate Simple Moving Average.

        Args:
            df: DataFrame with price data
            window: Moving average period

        Returns:
            Series containing SMA values with name format 'SMA{window}{timeframe}'
        """
        self._validate_dataframe(df)
        out = df.copy()
        tf = self.derive_timeframe(df)
        col_name = f'SMA{window}{tf}'
        out[col_name] = df['close'].rolling(window=window).mean()
        return out[col_name]

    def wma(self, df: pd.DataFrame, window: int) -> pd.Series:
        """Calculate Weighted Moving Average.

        Applies linearly increasing weights to more recent prices.

        Args:
            df: DataFrame with price data
            window: Moving average period

        Returns:
            Series containing WMA values with name format 'WMA{window}{timeframe}'
        """
        self._validate_dataframe(df)
        out = df.copy()
        weights = np.arange(1, window + 1)
        tf = self.derive_timeframe(df)
        col_name = f'WMA{window}{tf}'
        out[col_name] = df['close'].rolling(window=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(),
            raw=True
        )
        return out[col_name]

    def kama(self, df: pd.DataFrame, window: int, pow1: int = 2, pow2: int = 30) -> pd.Series:
        """Calculate Kaufman Adaptive Moving Average (KAMA).

        KAMA adjusts its smoothing based on market efficiency ratio.

        Args:
            df: DataFrame with price data
            window: Efficiency ratio period
            pow1: Fast EMA constant (default: 2)
            pow2: Slow EMA constant (default: 30)

        Returns:
            Series containing KAMA values with name format 'KAMA{window}'
        """
        self._validate_dataframe(df)
        out = df.copy()
        try:
            price = df['close']
            n = window
            absDiffx = abs(price - price.shift(1))  
            ER_num = abs(price - price.shift(n))
            ER_den = absDiffx.rolling(n).sum()
            ER = ER_num / ER_den
            sc = (ER * (2.0/(pow1+1)-2.0/(pow2+1.0))+2/(pow2+1.0)) ** 2.0
            answer = np.zeros(sc.size)
            N = len(answer)
            first_value = True
            for i in range(N):
                if sc.iloc[i] != sc.iloc[i]:
                    answer[i] = np.nan
                else:
                    if first_value:
                        answer[i] = price.iloc[i]
                        first_value = False
                    else:
                        answer[i] = answer[i-1] + sc.iloc[i] * (price.iloc[i] - answer[i-1])
            tf = self.derive_timeframe(df)
            col_name = f'KAMA{window}{tf}'
            out[col_name] = answer
            return out[col_name]
        except Exception as e:
            logger.error(f"Error calculating KAMA: {str(e)}")

    def ribbon(self, df: pd.DataFrame, ma: str = 'sma') -> pd.DataFrame:
        """Generate moving average ribbon.

        Creates multiple MAs with different periods to form a ribbon.

        Args:
            df: DataFrame with price data
            ma: Type of moving average ('sma', 'ema', 'wma', 'kama')

        Returns:
            DataFrame with original data plus MA columns

        Raises:
            ValueError: If invalid moving average type specified
        """
        self._validate_dataframe(df)
        ma_func = getattr(self, ma.lower(), None)
        if ma_func is None:
            raise ValueError(
                f"Invalid MA type '{ma}'. Available types: sma, ema, wma, kama"
            )

        df.columns = [x.lower() for x in df.columns]
        close = df['close'].to_frame()
        ma_series = [ma_func(close, window) for window in self.windows]
        out = pd.concat([close] + ma_series, axis=1)
        out.insert(0,'open', df['open'])
        out.insert(1,'high', df['high'])
        out.insert(2,'low', df['low'])
        out.insert(4,'volume', df['volume'])
        return out

    def concatenate_min_daily(self, min_df: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
        """Combine intraday and daily data.

        Args:
            min_df: DataFrame with intraday data
            daily_df: DataFrame with daily data

        Returns:
            DataFrame with combined data aligned to intraday index
        """
        min_df = min_df.copy()
        daily_df = daily_df.copy()
        min_df['day'] = min_df.index.date
        daily_df['day'] = daily_df.index.date
        cols = ['open', 'high', 'low', 'close', 'volume']
        daily_cols = [f'daily_{x}' for x in cols]
        daily_df.rename(columns=dict(zip(cols, daily_cols)), inplace=True)
        return pd.merge(
            min_df, daily_df, on='day', how='inner'
        ).drop(columns=['day']).set_index(min_df.index)

if __name__ == "__main__":
    import sys
    sys.path.append('/Users/jerald/Documents/Dir/Python/Stocks')
    from main import Pipeline as Manager
    from bin.main import get_path

    # Initialize
    connections = get_path()
    m = Manager(connections)
    ma = moving_avg()
    
    # Get sample data
    df = m.Pricedb.ohlc('aapl', daily=False, start="2025-01-10")
    daily_df = m.Pricedb.ohlc('aapl', daily=True)
    
    # Generate MA ribbons
    intraday_ribbon = ma.ribbon(df, ma='kama')
    daily_ribbon = ma.ribbon(daily_df, ma='ema')
    
    # Combine timeframes
    combined = ma.concatenate_min_daily(intraday_ribbon, daily_ribbon)
    
    # Display sample of results
    print("\nSample Analysis Results:")
    print("=======================")
    print(combined.tail())