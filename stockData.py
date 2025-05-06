# from bin.models.trends.trend_detector import TrendAnalyzer
# from bin.models.option_stats_model_setup import data as OptionsData
# from bin.models.indicator_model_setup import data as IndicatorData

from datetime import datetime, date
import pandas as pd
import re
import shutil
from pathlib import Path

from dataclasses import dataclass, field
from typing import Optional, Any, Dict

@dataclass(slots=True)
class StockData:
    stock: str
    manager: Any
    cache_dir: str = "data_cache"
    _price_data: Optional[pd.DataFrame] = field(init=False, default=None, repr=False, compare=False)
    _simulated_prices: Optional[pd.DataFrame] = field(init=False, default=None, repr=False, compare=False)
    _simulated_prices_cache: Optional[pd.DataFrame] = field(init=False, default=None, repr=False, compare=False)
    _indicators: Optional[pd.DataFrame] = field(init=False, default=None, repr=False, compare=False)
    _daily_option_stats: Optional[pd.DataFrame] = field(init=False, default=None, repr=False, compare=False)
    _option_chain: Optional[pd.DataFrame] = field(init=False, default=None, repr=False, compare=False)
    _daily_option_stats_cache: Dict[str, pd.DataFrame] = field(init=False, default_factory=dict, repr=False, compare=False)
    X: Optional[pd.DataFrame] = field(init=False, default=None, repr=False, compare=False)
    y: Optional[pd.DataFrame] = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self):
        Path(self.cache_dir).mkdir(exist_ok=True)

    def _load_or_cache(self, data_type: str, fetch_func, cache_key: str = None) -> pd.DataFrame:
        """
        Load DataFrame from cache or database, validating index and freshness.
        
        Args:
            data_type (str): Type of data (e.g., 'daily_option_stats').
            fetch_func (callable): Function to fetch fresh data.
            cache_key (str, optional): Custom cache key for parameterized data.
        
        Returns:
            pd.DataFrame: Cached or fresh DataFrame.
        """
        cache_key = cache_key or data_type
        cache_file = Path(self.cache_dir) / f"{self.stock}_{cache_key}.parquet"
        
        # Try loading from cache
        if cache_file.exists():
            data = pd.read_parquet(cache_file)
            # Validate index type for daily_option_stats
            if data_type == "daily_option_stats":
                if not isinstance(data.index, pd.DatetimeIndex):
                    print(f"Invalid cache for {cache_key}: Expected datetime index. Reloading.")
                    cache_file.unlink()
                else:
                    # Check if cache includes today's data (or yesterday's for weekends)
                    today = date.today()
                    latest_date = data.index.max().date()
                    # Allow yesterday's data if today is a non-trading day (e.g., weekend)
                    is_trading_day = today.weekday() < 5  # Monday-Friday
                    if is_trading_day and latest_date < today:
                        print(f"Cache for {cache_key} outdated: Latest date {latest_date}. Reloading.")
                        cache_file.unlink()
                    else:
                        return data
            else:
                return data
        
        # Fetch fresh data
        data = fetch_func()
        # Validate fetched data
        if data_type == "daily_option_stats":
            if not isinstance(data.index, pd.DatetimeIndex):
                raise ValueError(f"Fetched {data_type} for {self.stock} does not have a datetime index")
            # Optionally validate latest date
            latest_date = data.index.max().date()
            today = date.today()
            if today.weekday() < 5 and latest_date < today:
                print(f"Warning: Fetched {data_type} for {self.stock} may be outdated (latest: {latest_date})")
        
        # Cache and return
        data.to_parquet(cache_file, index=True)
        return data

    @property
    def price_data(self) -> pd.DataFrame:
        if self._price_data is None:
            self._price_data = self._load_or_cache(
                "price_data", lambda: self.manager.Pricedb.ohlc(self.stock)
            )
            self._price_data = self._price_data.reset_index()
            self._price_data.columns = self._price_data.columns.str.lower()
        return self._price_data.set_index("date")

    @property
    def indicators(self) -> pd.DataFrame:
        if self._indicators is None:
            self._indicators = self._load_or_cache(
                "indicators", lambda: self.manager.Pricedb.get_daily_technicals(self.stock)
            )
        return self._indicators

    @property
    def daily_option_stats(self) -> pd.DataFrame:
        if self._daily_option_stats is None:
            self._daily_option_stats = self._load_or_cache(
                "daily_option_stats", lambda: self.manager.Optionsdb.get_daily_option_stats(self.stock)
            )
        return self._daily_option_stats

    def get_daily_option_stats(self, dropCols: str = None) -> pd.DataFrame:
        """
        Get daily_option_stats with optional column dropping based on regex.
        
        Args:
            dropCols (str, optional): Regex pattern to match columns to drop (e.g., 'vol|oi|iv').
        
        Returns:
            pd.DataFrame: Filtered or full daily_option_stats DataFrame.
        
        Notes:
            Filtered DataFrames are re-cached if the base daily_option_stats is updated.
        """
        cache_key = f"daily_option_stats_{dropCols}" if dropCols else "daily_option_stats"
        base_data = self.daily_option_stats  # Ensure base data is fresh
        
        # Check if cached filtered data is still valid
        if cache_key in self._daily_option_stats_cache:
            cached_data = self._daily_option_stats_cache[cache_key]
            if cached_data.index.equals(base_data.index):  # Check if index matches base data
                return cached_data
            else:
                print(f"Invalid cache for {cache_key}: Index mismatch. Recreating.")
        
        # Create filtered DataFrame
        data = base_data.copy()
        if dropCols:
            try:
                columns_to_drop = [col for col in data.columns if re.search(dropCols, col)]
                if not columns_to_drop:
                    print(f"Warning: No columns matched regex '{dropCols}'")
                data = data.drop(columns=columns_to_drop, errors="ignore")
            except re.error:
                raise ValueError(f"Invalid regex pattern: {dropCols}")

        self._daily_option_stats_cache[cache_key] = data
        return data

    @property
    def option_chain(self) -> pd.DataFrame:
        if self._option_chain is None:
            self._option_chain = self._load_or_cache(
                "option_chain", lambda: self.manager.Optionsdb._parse_change_db(self.stock)
            )
        return self._option_chain

    def clear_cache(self, disk: bool = False, stock_specific: bool = True) -> None:
        """
        Clear in-memory and optionally disk caches.
        
        Args:
            disk (bool): If True, delete Parquet files from cache_dir.
            stock_specific (bool): If True, only delete files for this stock. If False, delete all files.
        
        Raises:
            FileNotFoundError: If cache_dir doesn't exist and disk=True.
        """
        self._price_data = None
        self._indicators = None
        self._daily_option_stats = None
        self._option_chain = None
        self._daily_option_stats_cache.clear()
        if disk:
            if not Path(self.cache_dir).exists():
                raise FileNotFoundError(f"Cache directory {self.cache_dir} does not exist")
            if stock_specific:
                for cache_file in Path(self.cache_dir).glob(f"{self.stock}_*.parquet"):
                    cache_file.unlink()
            else:
                shutil.rmtree(self.cache_dir)
                Path(self.cache_dir).mkdir()

    def get_features(self, drop_columns: str = None) -> pd.DataFrame:
        """
        Base feature extraction for ML.
        
        Args:
            drop_columns (str, optional): Regex pattern to drop columns from daily_option_stats.
        
        Returns:
            pd.DataFrame: Combined DataFrame of daily_option_stats and price_data.
        """
        option_stat = self.get_daily_option_stats(dropCols=drop_columns)
        price_data = self.indicators
        option_stat = option_stat.resample('1D').sum()
        price_df = price_data.loc[option_stat.index[0]:]
        option_df = option_stat.loc[price_df.index[0]:]
        df = pd.merge(option_df, price_df, left_index=True, right_index=True)
        return df



if __name__ == "__main__":
    from main import Manager 
    import numpy as np 
    # Example usage
    manager = Manager()  # Your Manager class
    sd = StockData(stock="spy", manager=manager, cache_dir="data_cache")
    sd.clear_cache(disk=True, stock_specific=False)
    df = sd.get_features()
    x = df.drop(columns=["close", "open", "high", "low"])
    y = df["close"].pct_change().shift(-1)
    y = pd.Series(np.sign(y), index=y.index)
    y = y.dropna()
    x = x.loc[y.index]

    print(df)


    # from bin.models.anom.model import StackedAnomalyModel
    # Example usage of StackedAnomalyModel
    # model = StackedAnomalyModel()
    # preds = model.train(
    #     stock = sd.stock, 
    #     x = x,
    #     y = y,
    #     contamination=0.01
    # )
    # print(preds)
