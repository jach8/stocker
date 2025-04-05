from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache

@dataclass
class TimeSeriesData:
    """Data container for time series analysis."""
    trend: np.ndarray| pd.Series
    seasonal: np.ndarray| pd.Series
    residual: np.ndarray| pd.Series
    observed: np.ndarray| pd.Series
    def __post_init__(self):
        self.df = pd.DataFrame({
            'trend': self.trend,
            'seasonal': self.seasonal,
            'residual': self.residual,
            self.observed.name : self.observed
        })

class TrendAnalyzer(ABC):
    """Base class for time series trend decomposition."""
    
    def __init__(self, period: int = 7, model: str = 'additive'):
        """
        Initialize the trend analyzer.
        
        Args:
            period (int): Seasonality period (e.g., 7 for weekly patterns)
            model (str): Type of decomposition model ('additive' or 'multiplicative')
        """
        self.period = period
        self.model = model
        
    
    def decompose(self, data: np.ndarray) -> TimeSeriesData:
        """
        Decompose time series into trend, seasonal, and residual components.
        
        Args:
            data (np.ndarray): Time series data to decompose
            
        Returns:
            TimeSeriesData: Decomposed components
            
        Raises:
            ValueError: If data length is insufficient for decomposition
        """
        if len(data) < self.period * 2:
            raise ValueError(f"Data length must be at least {self.period * 2} points")
            
        result = seasonal_decompose(
            data,
            period=self.period,
            model=self.model,
            extrapolate_trend='freq'
        )
        
        return TimeSeriesData(
            trend=result.trend,
            seasonal=result.seasonal,
            residual=result.resid,
            observed=result.observed
        )
    
    def analyze_trend(self, data: np.ndarray) -> TimeSeriesData:
        """
        Analyze the trend of the time series data.
        
        Args:
            data (np.ndarray): Time series data to analyze
            
        Returns:
            TimeSeriesData: Decomposed components
        """
        tmp = self.decompose(data)
        self.data = tmp
        checks = ['6D', '28D', '96D']
        name = data.name
        # Report the trend for each check, (lookbac = check in checks)
        bull = 0; bear = 0
        for check in checks:
            last_trend = tmp.trend.resample(check).last().diff().tail(1).values[0]
            if last_trend > 0:
                print(f"{name.upper()} Increase {check}")
                bull += 1
            elif last_trend < 0:
                print(f"{name.upper()} Decrease {check}")
                bear += 1
            else:
                print(f"{name.upper()} Choppy {check}")
                if bull > bear:
                    bull += 1
                else:
                    bear += 1
        if bull > bear:
            return 'Increase'
        else:
            return 'Decrease'

        
