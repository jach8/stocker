"""
Module for detecting changes in stock price trends.
Provides functionality to analyze trends across multiple stocks using the TrendAnalyzer.
"""
import logging
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

from bin.models.trends.trend_detector import TrendAnalyzer
from bin.models.trends.change_detection import ChangePointDetector
from bin.models.trends.peakDetector import PeakDetector
from bin.main import get_path
from main import Manager

@dataclass
class TrendResult:
    """Container for trend analysis results."""
    stock: str
    name: str
    trend_direction: str
    seasonality: str
    slope: float
    change_point: Optional[float] = None
    valley: Any = None
    peaks: Any = None
    # Return as a dictionary
    def to_dict(self) -> Dict:
        return {
            "stock": self.stock,
            "name": self.name,
            "trend_direction": self.trend_direction,
            "seasonality": self.seasonality,
            "slope": self.slope,
            "change_point": self.change_point,
            "valley": self.valley,
            "peaks": self.peaks
        }

class TResults:
    """Class for detecting and analyzing changes in stock price trends."""
    
    def __init__(self, connections: Dict|str, lookback_days: int = 90, window_size: int = 30, period: int = 3):
        """
        Initialize the TResults.
    
        Args:
            connections: Dictionary of database connections
            lookback_days: Number of days to look back for trend analysis
            window_size: Size of the rolling window for trend analysis
            period: Period for trend analysis
        """
        self.lookback_days = lookback_days
        self.window_size = window_size
        self.period = period
        
        # Initialize data connection
        self.data_manager = Manager(connections)
        self.stocks = self.data_manager.Pricedb.stocks['all_stocks']
        self.trend_analyzer = TrendAnalyzer(period = self.period)
        self.peak_detector = PeakDetector(prominence=0.5, distance = 2)

    def get_aligned_data(self, stock: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get aligned data for a given stock.

        Args:
            stock: Stock symbol to analyze

        Returns:
            Tuple of DataFrames containing aligned stock and option data
        """
        ohlcv = self.data_manager.Pricedb.ohlc(stock).dropna().tail(self.lookback_days)
        option_db = self.data_manager.Optionsdb.get_daily_option_stats(stock).dropna()
        if ohlcv.empty or option_db.empty:
            raise ValueError(f"No data available for {stock}")
        
        return ohlcv, option_db

    def analyze_single_stock(self, stock: str) -> Optional[List[TrendResult]]:
        """
        Analyze trends for a single stock across multiple metrics.

        Args:
            stock: Stock symbol to analyze

        Returns:
            List of TrendResult objects if analysis successful, None otherwise
        
        Raises:
            ValueError: If required data is missing or invalid
        """
        try:
            # Get and validate data
            ohlcv, option_db = self.get_aligned_data(stock)
            
            # Define metrics to analyze
            metrics = {
                'close_prices': ohlcv['Close'],
                'stock_volume': ohlcv['Volume'],
                'options_volume': option_db['total_vol'],
                'oi': option_db['total_oi'],
                'atm_iv': option_db['atm_iv'], 
                'call_oi': option_db['call_oi'],
                'put_oi': option_db['put_oi'],
                'call_volume': option_db['call_vol'],
                'put_volume': option_db['put_vol'],
            }
            
            results = []
            for metric_name, data in metrics.items():
                # Analyze trends
                trend_direction, seasonality, slope = self.trend_analyzer.analyze(data)
                
                # Configure change point detection

                window_size = self.window_size
                scale = True
                period = self.period
                
                # Detect change points
                signal = ChangePointDetector(
                    data,
                    scale=scale,
                    period=period,
                    window_size=window_size
                )
                change_points = signal.get_last_change_point(
                    # Sensitivity from 0.01, to 0.2, with a step of 0.04
                    sensitivity_range=(0.01, 0.2, 0.04), 
                    # Threshold from 0.5 to 2, with a step of 0.5
                    threshold_range=(0.5, 2, 0.5),
                    min_triggers = 4, 
                    max_triggers = 20
                    ).Signal


                peaks = self.peak_detector.find_peaks(data.values)
                peak_dates = data.index[peaks]
                valleys = self.peak_detector.find_valleys(data.values)
                valley_dates = data.index[valleys]
                
                # Create and store result
                results.append(
                    TrendResult(
                        stock=stock,
                        name=metric_name,
                        trend_direction=trend_direction,
                        seasonality=seasonality,
                        slope=slope,
                        change_point=change_points,
                        valley=valley_dates.max(),
                        peaks=peak_dates.max(),
                    )
                )
            
            return results

        except ValueError as ve:
            logging.error(f"Data validation error for {stock}: {str(ve)}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error analyzing {stock}: {str(e)}")
            return None
        
    def __convert_to_dataframe(self, results: List[TrendResult]) -> pd.DataFrame:
        """
        Convert list of TrendResult objects to a DataFrame.

        Args:
            results: List of TrendResult objects

        Returns:
            DataFrame containing trend analysis results
        """
        # Convert to DataFrame
        data = []
        for i in results:
            for result in i:
                data.append({
                    'stock': result.stock,
                    'metric': result.name,
                    'trend_direction': result.trend_direction,
                    'seasonality': result.seasonality,
                    'slope': result.slope,
                    'change_point': result.change_point, 
                    'last_valley': result.valley,
                    'last_peak': result.peaks,
                })
        df = pd.DataFrame(data)

        # Flag slope discrepancies (where trend direction and slope sign don't match)
        df['slope_discrepancy'] = ((df['trend_direction'] == 'up') & (df['slope'] < 0)) | ((df['trend_direction'] == 'down') & (df['slope'] > 0))
        return df 

    def analyze_stocks(self, stocks: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Analyze trends for multiple stocks.

        Args:
            stocks: List of stock symbols to analyze. If None, uses all stocks.

        Returns:
            DataFrame containing trend analysis results
        """
        stocks_to_analyze = stocks if stocks is not None else self.stocks
        results = []
        
        pbar = tqdm(stocks_to_analyze, desc='Analyzing Stocks')
        for stock in pbar:
            pbar.set_description(f'Processing {stock}')
            result = self.analyze_single_stock(stock)
            if result:
                results.append(result)
                pbar.set_postfix({'Success': True})


        # Convert results to DataFrame
        self.result_df = self.__convert_to_dataframe(results)
        return results
    
def main():
    """Example usage of the TResults class."""
    connections = get_path()
    detector = TResults(connections, lookback_days=15)
    results = detector.analyze_stocks()
    print("\nTrend Analysis Results:")
    print(results)

    print(detector.result_df)
    detector.result_df.to_csv("trend_analysis_results.csv", index=False)


if __name__ == "__main__":
    main()
