"""Chart Analysis and Signal Generation Module

This module provides tools for technical analysis and signal generation
based on price action and moving averages.
"""

import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
import re
import os
import sys
from typing import Optional, Dict, Any, Tuple, NamedTuple

# Import moving_avg class
from ma import moving_avg

class MAParams(NamedTuple):
    """Parameters for moving average calculation."""
    type: str  # 'EMA', 'SMA', etc.
    period: int  # Number of periods
    timeframe: Optional[str] = None  # 'D', 'W', 'M', etc.

class SignalGenerationError(Exception):
    """Exception raised for errors in signal generation."""
    pass

class TimeframeError(Exception):
    """Exception raised for invalid timeframe specifications."""
    pass

class ChartAnalyzer(moving_avg):
    """Analyze price charts and generate trading signals.
    
    This class extends moving_avg to add sophisticated trend analysis
    and signal generation capabilities with timeframe-aware indicators.
    """
    
    # Valid timeframe suffixes and their pandas resampling strings
    VALID_TIMEFRAMES = {
        'T': 'min',   # Minute
        'H': 'H',     # Hour
        'D': 'D',     # Day
        'W': 'W',     # Week
        'M': 'M'      # Month
    }
    
    def __init__(self, 
                momentum_window: int = 14,
                consolidation_threshold: float = 0.02,
                trend_threshold: float = 0.01):
        """Initialize chart analyzer with signal generation parameters."""
        super().__init__()
        self.momentum_window = momentum_window
        self.consolidation_threshold = consolidation_threshold
        self.trend_threshold = trend_threshold

    def _calculate_momentum(self, df: pd.DataFrame) -> pd.Series:
        """Calculate price momentum using rate of change.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Series with momentum values
        """
        self._validate_dataframe(df)
        roc = df['close'].pct_change(self.momentum_window)
        return roc

    def _resample_and_calculate_ma(self, df: pd.DataFrame, params: MAParams) -> pd.Series:
        """Calculate MA with proper timeframe resampling.
        
        Args:
            df: Price DataFrame
            params: Moving average parameters
            
        Returns:
            Series containing MA values
        """
        if params.timeframe and params.timeframe in self.VALID_TIMEFRAMES:
            # Resample data to desired timeframe
            resample_rule = self.VALID_TIMEFRAMES[params.timeframe]
            resampled = df.resample(resample_rule).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            # Calculate MA on resampled data
            ma_func = getattr(self, params.type)
            ma_series = ma_func(resampled[['close']], params.period)
            
            # Align with original index
            return ma_series.reindex(df.index).fillna(method='ffill')
        else:
            # Calculate MA directly if no timeframe specified
            ma_func = getattr(self, params.type)
            return ma_func(df[['close']], params.period)

    def _detect_highs_lows(self, df: pd.DataFrame, window: int = 20) -> Tuple[pd.Series, pd.Series]:
        """Detect higher highs and lower lows in price action.
        
        Args:
            df: DataFrame with price data
            window: Lookback window for comparisons
            
        Returns:
            Tuple containing:
            - Series indicating higher highs (1 if detected, 0 otherwise)
            - Series indicating lower lows (1 if detected, 0 otherwise)
        """
        self._validate_dataframe(df)
        
        # Rolling max/min for comparison
        rolling_high = df['high'].rolling(window=window).max()
        rolling_low = df['low'].rolling(window=window).min()
        
        # Detect new highs/lows
        higher_highs = (df['high'] > rolling_high.shift(1)).astype(int)
        lower_lows = (df['low'] < rolling_low.shift(1)).astype(int)
        
        return higher_highs, lower_lows

    def _calculate_ma_convergence(self, df: pd.DataFrame,
                              fast_ma: str = 'EMA20',
                              slow_ma: str = 'EMA50') -> pd.Series:
        """Calculate moving average convergence/divergence.
        
        Args:
            df: DataFrame with price and MA data
            fast_ma: Column name of faster MA
            slow_ma: Column name of slower MA
            
        Returns:
            Series indicating convergence (positive) or divergence (negative)
        """
        if fast_ma not in df.columns or slow_ma not in df.columns:
            raise SignalGenerationError(f"Required MAs not found: {fast_ma}, {slow_ma}")
            
        # Calculate distance between MAs
        ma_distance = (df[fast_ma] - df[slow_ma]) / df[slow_ma]
        
        # Calculate rate of change in distance
        convergence = -ma_distance.diff()  # Negative diff means MAs are converging
        
        return convergence

    def _calculate_trend_slope(self, series: pd.Series, window: int = 20) -> pd.Series:
        """Calculate trend slope using linear regression.
        
        Args:
            series: Price or MA series
            window: Window for slope calculation
            
        Returns:
            Series of slope values
        """
        slopes = pd.Series(index=series.index, dtype=float)
        
        for i in range(window, len(series)):
            y = series.iloc[i-window:i]
            x = np.arange(window)
            slope, _ = np.polyfit(x, y, 1)
            slopes.iloc[i] = slope
            
        return slopes

    def _validate_timeframe(self, timeframe: Optional[str]) -> None:
        """Validate timeframe suffix.
        
        Args:
            timeframe: Timeframe indicator ('D', 'W', etc.)
            
        Raises:
            TimeframeError: If timeframe is invalid
        """
        if timeframe and timeframe not in self.VALID_TIMEFRAMES:
            raise TimeframeError(
                f"Invalid timeframe: {timeframe}. "
                f"Valid timeframes: {list(self.VALID_TIMEFRAMES.keys())}"
            )

    def _parse_ma_name(self, ma_name: str) -> MAParams:
        """Parse moving average name into parameters.
        
        Args:
            ma_name: Moving average name (e.g., 'EMA20D')
            
        Returns:
            MAParams tuple with type, period, and timeframe
        """
        match = re.match(r'([A-Z]+)(\d+)([A-Z]*)', ma_name)
        if not match:
            raise ValueError("Invalid MA name format")
        
        ma_type, ma_period, ma_timeframe = match.groups()
        return MAParams(ma_type, int(ma_period), ma_timeframe or None)

    def generate_signals(self, df: pd.DataFrame, 
                        params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Generate trading signals based on technical analysis.
        
        This method supports timeframe-specific moving averages:
        - 'EMA20D': 20-day Exponential Moving Average
        - 'SMA50W': 50-week Simple Moving Average
        - 'WMA10': 10-period Weighted Moving Average (uses data as-is)
        
        Args:
            df: DataFrame with OHLCV data
            params: Optional parameters dict with keys:
                - fast_ma: Fast MA specification (default: 'EMA20D')
                - slow_ma: Slow MA specification (default: 'EMA50D')
                - momentum_threshold: Min momentum for trend confirmation
                
        Returns:
            DataFrame with signals and indicators
        """
        self._validate_dataframe(df)
        
        # Use provided params or defaults
        params = params or {}
        fast_ma = params.get('fast_ma', 'EMA20D')
        slow_ma = params.get('slow_ma', 'EMA50D')
        momentum_threshold = params.get('momentum_threshold', 0.01)
        
        # Parse MA specifications
        try:
            fast_params = self._parse_ma_name(fast_ma)
            slow_params = self._parse_ma_name(slow_ma)
            
            # Validate timeframes
            self._validate_timeframe(fast_params.timeframe)
            self._validate_timeframe(slow_params.timeframe)
            
        except (ValueError, TimeframeError) as e:
            raise SignalGenerationError(f"Invalid MA specification: {str(e)}")
        
        # Calculate MAs with proper timeframe handling using ribbon
        df = df.copy()
        original_index = df.index.copy()  # Store original index
        
        # Ensure index is sorted
        df = df.sort_index()
        
        # Generate EMA ribbon
        df = self.ribbon(df, ma='ema')
        
        # Verify index preservation
        if not df.index.equals(original_index):
            logger.warning("Index changed during signal generation. Realigning to original index...")
            df = df.reindex(original_index)
        
        # Get actual column names from ribbon output matching requested periods
        fast_period = fast_params.period
        slow_period = slow_params.period
        tf = self.derive_timeframe(df)
        
        actual_fast_ma = f"{fast_params.type}{fast_period}{tf}"
        actual_slow_ma = f"{slow_params.type}{slow_period}{tf}"
        
        # Debug: Print expected column names
        print("\nLooking for MA columns:")
        print(f"Fast MA: {actual_fast_ma}")
        print(f"Slow MA: {actual_slow_ma}")
        
        # Calculate technical indicators using actual column names
        momentum = self._calculate_momentum(df)
        higher_highs, lower_lows = self._detect_highs_lows(df)
        convergence = self._calculate_ma_convergence(df, actual_fast_ma, actual_slow_ma)
        price_slope = self._calculate_trend_slope(df['close'])
        
        # Generate signals
        signals = pd.Series(0, index=df.index)
        
        # Buy signals
        buy_conditions = (
            (convergence > 0) &  # MAs are converging
            (momentum > momentum_threshold) &  # Positive momentum
            (higher_highs == 1) &  # New highs forming
            (price_slope > self.trend_threshold)  # Upward trend
        )
        signals[buy_conditions] = -1
        
        # Sell signals
        sell_conditions = (
            (momentum < -momentum_threshold) &  # Negative momentum
            (lower_lows == 1) &  # New lows forming
            (price_slope < -self.trend_threshold)  # Downward trend
        )
        signals[sell_conditions] = 1
        
        # Add signal and indicator columns
        df['signal'] = signals
        df['trend_strength'] = price_slope
        df['momentum'] = momentum
        df['ma_convergence'] = convergence
        
        return df

    def plot_signals(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """Plot price chart with signals and indicators."""
        if 'signal' not in df.columns:
            raise SignalGenerationError("No signals found. Run generate_signals first.")
        
        # Create subplot configuration
        subplots = [
            ('momentum', 0.2),
            ('ma_convergence', 0.2),
            ('trend_strength', 0.2)
        ]
        
        # Get MA columns for overlay
        ma_columns = [col for col in df.columns if any(
            ma in col.upper() for ma in ['EMA', 'SMA', 'WMA', 'KAMA']
        )]
        
        # Create MA overlays
        ma_colors = ['blue', 'red', 'green', 'purple', 'orange']
        ma_plots = [
            mpf.make_addplot(
                df[col],
                color=ma_colors[i % len(ma_colors)],
                width=1,
                alpha=0.7
            ) for i, col in enumerate(ma_columns)
        ]
        
        # Create indicator subplots
        indicator_plots = [
            mpf.make_addplot(df[col], panel=i+1, ylabel=col)
            for i, (col, _) in enumerate(subplots)
        ]

        # Create buy/sell signal markers
        buy_markers = df['close'].copy()
        sell_markers = df['close'].copy()
        
        # Set values for markers only at signal points
        buy_markers[df['signal'] != 1] = np.nan
        sell_markers[df['signal'] != -1] = np.nan
        
        # # Create buy/sell plots
        # buy_plot = mpf.make_addplot(buy_markers, type='scatter', markersize=1000,
        #                           marker='^', color='g', panel=0)
        # sell_plot = mpf.make_addplot(sell_markers, type='scatter', markersize=1000,
        #                            marker='v', color='r', panel=0)
        
        # Create plot with indicators
        fig, axlist = mpf.plot(
            df,
            type='candle',
            volume=False,
            figsize=(15, 10),
            style='yahoo',
            addplot=ma_plots,
            returnfig=True,
            show_nontrading=True
        )
        
        # Plot the buy signals
        axlist[0].scatter(df.index, buy_markers, marker='o', color='g')


        # Plot the sell signal
        axlist[0].scatter(df.index, sell_markers, marker='o', color='r')

        
        # Add legend for MAs
        if ma_columns:
            axlist[0].legend(ma_columns)
        
        # if save_path:
        #     plt.savefig(save_path, bbox_inches='tight')
        plt.show()
        
        plt.close()

if __name__ == "__main__":
    import sys
    sys.path.append('/Users/jerald/Documents/Dir/Python/Stocks')
    from main import Pipeline as Manager
    from bin.main import get_path
    
    # Initialize
    connections = get_path()
    m = Manager(connections)
    analyzer = ChartAnalyzer()
    
    # Get sample data
    df = m.Pricedb.ohlc('nvda', daily=True, start="2000-01-01")
    
    # Generate signals with timeframe-specific MAs
    signals = analyzer.generate_signals(
        df,
        params={
            'fast_ma': 'EMA20D',  # 20-day EMA
            'slow_ma': 'EMA50D',  # 50-day EMA
            'momentum_threshold': 0.01
        }
    )
    
    # Display results
    print("\nSignal Analysis Results:")
    print("=======================")
    print(signals.tail())
    
    # Plot results with MA overlay
    analyzer.plot_signals(signals, save_path='aapl_signals.png')
    print("\nChart saved as 'aapl_signals.png'")