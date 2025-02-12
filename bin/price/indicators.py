import pandas as pd 
import numpy as np 
import sqlite3 as sql
import logging
from typing import Union, Tuple, Dict, List, Optional
from pandas import Series, DataFrame

# Configure logging
logging.basicConfig(
    filename='logs/indicators.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Indicators:
    def __init__(self, price: Optional[Union[pd.Series, pd.DataFrame]] = None):
        ''' Indicator Class to compute technical Indicators 
            Inputs: 
                - price: pd.Series or pd.DataFrame: containing the price data, can be OHLCV, or just the Close. 
            
            Methods:
                - EMA: Exponential Moving Average
                - sma: Simple Moving Average
                - macd: Moving Average Convergence Divergence
                - ATR: Average True Range
                - ADX: Average Directional Index
                - BB: Bollinger Bands
                - stochastic: Stochastic Oscillator
                - slow_stoch: Slow Stochastic Oscillator
                - momentum: Momentum
                - LOI: High Probability Price Levels
                - keltner: Keltner Channels
                - KAMA: Kaufman's Adaptive Moving Average
                - rsi: Relative Strength Index
                - get_indicators: Returns a dictionary of indicators
                - indicator_df: Returns a dataframe of indicators
                - _get_moving_averages: Returns a dictionary of moving averages
                - _get_volatility: Returns a dictionary of volatility indicators
                - _get_momentum: Returns a dictionary of momentum indicators
                
        '''
        self.is_df = False
        self.price = None
        self.high = None
        self.low = None
        self.open = None
        self.volume = None
        self.dte_index = None
        
        if price is not None:
            try:
                self.fit(price)
            except Exception as e:
                logger.error(f"Error initializing Indicators: {str(e)}")
                raise
            
    def fit(self, price: Union[pd.Series, pd.DataFrame]) -> None:
        ''' Fit the price data. '''
        try:
            if isinstance(price, pd.DataFrame):
                price.columns = [x.lower() for x in price.columns]  
                self.price = price['close']
                self.high = price['high']
                self.low = price['low']
                self.open = price['open']
                self.volume = price['volume']
                self.dte_index = price.index
                self.is_df = True
                self.get_indicators()
            else:
                self.price = price
                self.dte_index = price.index
        except Exception as e:
            logger.error(f"Error fitting price data: {str(e)}")
            raise
            
    def est_vol(self, lookback: int = 10) -> np.ndarray:
        """ 
        This is the Yang-Zheng (2005) Estimator for Volatility; 
            Yang Zhang is a historical volatility estimator that handles 
                1. opening jumps
                2. the drift and has a minimum estimation error 
        """
        try:
            o = self.open
            h = self.high
            l = self.low
            c = self.price
            k = 0.34 / (1.34 + (lookback+1)/(lookback-1))
            cc = np.log(c/c.shift(1))
            ho = np.log(h/o)
            lo = np.log(l/o)
            co = np.log(c/o)
            oc = np.log(o/c.shift(1))
            oc_sq = oc**2
            cc_sq = cc**2
            rs = ho*(ho-co)+lo*(lo-co)
            close_vol = cc_sq.rolling(window=lookback).sum() * (1.0 / (lookback - 1.0))
            open_vol = oc_sq.rolling(window=lookback).sum() * (1.0 / (lookback - 1.0))
            window_rs = rs.rolling(window=lookback).sum() * (1.0 / (lookback - 1.0))
            result = (open_vol + k * close_vol + (1-k) * window_rs).apply(np.sqrt) * np.sqrt(252)
            result[:lookback-1] = np.nan
            return result
        except Exception as e:
            logger.error(f"Error calculating volatility: {str(e)}")
            raise

    def EMA(self, price: Union[pd.Series, np.ndarray], window: int) -> pd.Series:
        ''' (Wilder's) Exponential Moving Average. '''
        try:
            price = pd.Series(price)
            return price.ewm(alpha=1/window).mean()
        except Exception as e:
            logger.error(f"Error calculating EMA: {str(e)}")
            raise

    def ema(self, window: int) -> np.ndarray:
        ''' (Wilder's) Exponential Moving Average. '''
        try:
            return np.array(self.price.ewm(span=window).mean())
        except Exception as e:
            logger.error(f"Error calculating EMA: {str(e)}")
            raise
    
    def sma(self, window: int) -> np.ndarray:
        ''' Simple Moving Average. '''
        try:
            return np.array(self.price.rolling(window=window).mean())
        except Exception as e:
            logger.error(f"Error calculating SMA: {str(e)}")
            raise
    
    def macd(self, fast_window: int = 10, slow_window: int = 20) -> Tuple[np.ndarray, pd.Series]:
        ''' Moving Average Convergence Divergence. '''
        try:
            mcd = self.ema(fast_window) - self.sma(slow_window)
            mcd_signal = pd.Series(mcd, index=self.dte_index).ewm(span=9).mean()
            return mcd, mcd_signal
        except Exception as e:
            logger.error(f"Error calculating MACD: {str(e)}")
            raise
    
    def ATR(self, window: int) -> np.ndarray:
        ''' Average True Range.: Need three columns, High, Low, and Close. '''
        try:
            if self.high is None or self.low is None:
                hi = self.price.rolling(window=window).min().values
                lo = self.price.rolling(window=window).max().values
            else: 
                hi = self.high.rolling(window=window).min().values
                lo = self.low.rolling(window=window).max().values
            c = self.price.values
            tr = np.vstack([np.abs(hi[1:]-c[:-1]),np.abs(lo[1:]-c[:-1]),(hi-lo)[1:]]).max(axis=0)
            tr = np.concatenate([[np.nan], tr])
            return self.EMA(tr, window).values
        except Exception as e:
            logger.error(f"Error calculating ATR: {str(e)}")
            raise

    def ADX(self, window: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ''' Average Directional Index. '''
        try:
            if self.high is None or self.low is None: 
                hi = self.price.rolling(window=window).min().values 
                lo = self.price.rolling(window=window).max().values 
            else: 
                hi = self.high.rolling(window=window).min().values 
                lo = self.low.rolling(window=window).max().values
            c = self.price.values 
            up = hi[1:] - hi[:-1]
            down = lo[:-1] - lo[1:]
            up_ind = up > down
            down_ind = down > up
            dmup = np.zeros(len(up))
            dmdown = np.zeros(len(down))
            dmup = np.where(up_ind, up, 0)
            dmdown = np.where(down_ind, down, 0)
            atr = self.ATR(window)[1:]
            diplus = 100 * self.EMA(dmup, window) / atr
            diminus = 100 * self.EMA(dmdown, window) / atr
            diminus[(diplus + diminus) == 0] = 1e-5
            dx = 100 * np.abs(diplus - diminus) / (diplus + diminus)
            dx = np.concatenate([[np.nan], dx])
            return self.EMA(dx, window).values, diplus.values, diminus.values
        except Exception as e:
            logger.error(f"Error calculating ADX: {str(e)}")
            raise

    def BB(self, window: int = 20, m: float = 2) -> np.ndarray:
        ''' Bollinger Bands. '''
        try:
            sma = self.price.rolling(window=window).mean()
            sigma = self.price.rolling(window=window).std()
            return np.array((self.price - sma) / (m * sigma))
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            raise

    def stochastic(self, window: int = 14) -> np.ndarray:
        ''' Stochastic Oscillator. '''
        try:
            h14 = self.price.rolling(window=window).max()
            l14 = self.price.rolling(window=window).min()
            return np.array((self.price - l14) / (h14 - l14))
        except Exception as e:
            logger.error(f"Error calculating Stochastic Oscillator: {str(e)}")
            raise
    
    def slow_stoch(self, window: int = 14) -> np.ndarray:
        ''' Slow Stochastic Oscillator. '''
        try:
            fast = pd.Series(self.stochastic(window))
            return fast.rolling(window=3).mean().values
        except Exception as e:
            logger.error(f"Error calculating Slow Stochastic: {str(e)}")
            raise

    def momentum(self, window: int = 10) -> np.ndarray:
        ''' Momentum. '''
        try:
            return np.array((self.price - self.price.shift(window)) / self.price.shift(window))
        except Exception as e:
            logger.error(f"Error calculating Momentum: {str(e)}")
            raise

    def LOI(self, window: Optional[int] = None, out: int = 5) -> np.ndarray:
        ''' Return High Probability Price Levels, for a given window. '''
        try:
            if window is None: 
                window = len(self.price)
            p = self.price.resample('1min').last()
            x, y = np.unique(p.tail(window), return_counts=True)
            y = y / len(x)
            return x[-out:]
        except Exception as e:
            logger.error(f"Error calculating LOI: {str(e)}")
            raise

    def keltner(self, window: int = 20, m: float = 2) -> np.ndarray:
        ''' Keltner Channels. Indicator '''
        try:
            return np.array((self.price - self.ema(window)) / (m * self.ATR(window)))
        except Exception as e:
            logger.error(f"Error calculating Keltner Channels: {str(e)}")
            raise
    
    def KAMA(self, n: int = 10, pow1: float = 2, pow2: float = 30) -> np.ndarray:
        ''' kama indicator '''    
        try:
            price = self.price
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
            return answer
        except Exception as e:
            logger.error(f"Error calculating KAMA: {str(e)}")
            raise
    
    def rsi(self, window: int = 14) -> np.ndarray:
        ''' Relative Strength Index. '''
        try:
            delta = self.price.diff()
            up_days = delta.copy()
            up_days[delta<=0]=0.0
            down_days = abs(delta.copy())
            down_days[delta>0]=0.0
            RS_up = up_days.rolling(window).mean()
            RS_down = down_days.rolling(window).mean()
            out = (100-100/(1+RS_up/RS_down))
            return out.values / 100
        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}")
            raise

    def mean_reversion(self, window: int = 20) -> np.ndarray:
        ''' Mean Reversion Indicator. '''
        try:
            return np.array(self.price.rolling(window).mean() - self.price / self.price.rolling(window).std())
        except Exception as e:
            logger.error(f"Error calculating Mean Reversion: {str(e)}")
            raise
    
    def mean_reversion_z(self, window: int = 20) -> np.ndarray:
        """ Mean reverersion z-score, any values over 1.5 are considered overbought, and values under -1.5 are considered oversold. 
            returns an array where 1 indicates overbought, -1 indicates oversold, and 0 indicates neutral.
        """
        try:
            z = self.mean_reversion(window)
            return np.where(z > 1.5, 1, np.where(z < -1.5, -1, 0))
        except Exception as e:
            logger.error(f"Error calculating Mean Reversion Z-Score: {str(e)}")
            raise

    def get_indicators(self, fast: int = 10, medium: int = 14, 
                      slow: int = 35, m: float = 2) -> Dict[str, np.ndarray]:
        try:
            adx, diplus, diminus = self.ADX(medium)
            macd, macd_signal = self.macd(fast, medium)
            d = {
                'ema_fast': self.ema(fast),
                'ema_med': self.ema(medium),
                'ema_slow': self.ema(slow),
                'sma_med': self.sma(medium),
                'sma_slow': self.sma(slow),
                'sma_fast': self.sma(fast),
                'kama_fast': self.KAMA(n=fast, pow1=m, pow2=medium),
                'kama_med': self.KAMA(n=medium, pow1=m, pow2=slow),
                'kama_slow': self.KAMA(n=slow, pow1=m, pow2=slow * 1.5),
                'macd': macd,
                'macd_signal': macd_signal,
                'atr': self.ATR(fast),
                'adx': adx,
                'fast_bb': self.BB(fast, m),
                'slow_bb': self.BB(medium, m),
                'fast_kc': self.keltner(fast, m),
                'slow_kc': self.keltner(medium, m),
                'fast_stoch': self.stochastic(fast),
                'slow_stoch': self.slow_stoch(fast),       
                'mom': self.momentum(medium),
                'rsi': self.rsi(medium),
                'z_score': self.mean_reversion(medium),
                'volatility': self.est_vol(lookback=fast)
            }
            if self.is_df == True:
                d['Open'] = np.array(self.open)
                d['High'] = np.array(self.high)
                d['Low'] = np.array(self.low)
                d['Close'] = np.array(self.price)
                d['Volume'] = np.array(self.volume)
            return d
        except Exception as e:
            logger.error(f"Error getting indicators: {str(e)}")
            raise
        
    def get_levels(self, fast: int, medium: int, slow: int, out: int = 5) -> Dict[str, np.ndarray]:
        try:
            return {'levels': self.LOI()}
        except Exception as e:
            logger.error(f"Error getting levels: {str(e)}")
            raise
    
    def indicator_df(self, fast: int = 10, medium: int = 14, 
                    slow: int = 35, m: float = 2) -> pd.DataFrame:
        ''' Return a dataframe of indicators. '''
        try:
            d = self.get_indicators(fast, medium, slow, m)
            out = pd.DataFrame(d, index=self.dte_index)
            out.columns = [x.lower() for x in out.columns]
            self.states = self.get_states(fast, medium, slow, m)
            return out
        except Exception as e:
            logger.error(f"Error creating indicator DataFrame: {str(e)}")
            raise
    
    def _get_moving_averages(self, fast: int = 10, medium: int = 14, 
                           slow: int = 35, m: float = 2) -> Dict[str, np.ndarray]:
        ''' Return a dictionary of moving averages. '''
        try:
            return {
                'ema_fast': self.ema(fast),
                'ema_med': self.ema(medium),
                'ema_slow': self.ema(slow),
                'sma_med': self.sma(medium),
                'sma_slow': self.sma(slow),
                'sma_fast': self.sma(fast),
                'kama_fast': self.KAMA(n=fast, pow1=m, pow2=medium),
                'kama_med': self.KAMA(n=medium, pow1=m, pow2=slow),
                'kama_slow': self.KAMA(n=slow, pow1=m, pow2=slow * 1.5),
            }
        except Exception as e:
            logger.error(f"Error getting moving averages: {str(e)}")
            raise
        
    def _get_volatility(self, fast: int = 10, medium: int = 14, 
                       slow: int = 35, m: float = 2) -> Dict[str, np.ndarray]:
        ''' Return a dictionary of volatility indicators. '''
        try:
            return {
                'volatility': self.est_vol(lookback=fast),
                'atr': self.ATR(fast),
                'adx': self.ADX(slow),
                'fast_bb': self.BB(fast, m),
                'slow_bb': self.BB(medium, m),
                'fast_kc': self.keltner(fast, m),
                'slow_kc': self.keltner(medium, m),
            }
        except Exception as e:
            logger.error(f"Error getting volatility indicators: {str(e)}")
            raise
        
    def _get_momentum(self, fast: int = 10, medium: int = 14, 
                     slow: int = 35, m: float = 2) -> Dict[str, np.ndarray]:
        ''' Return a dictionary of momentum indicators. '''
        try:
            return {
                'fast_stoch': self.stochastic(fast),
                'slow_stoch': self.slow_stoch(fast),
                'mom': self.momentum(medium),
                'rsi': self.rsi(medium),
            }
        except Exception as e:
            logger.error(f"Error getting momentum indicators: {str(e)}")
            raise
    
    def get_states(self, fast: int = 10, medium: int = 14, 
                  slow: int = 35, m: float = 2) -> pd.DataFrame:
        """
        Returns the action states of the indicators. 
            ema_fm: EMA(fast) - EMA(medium)
            ema_ms: EMA(medium) - EMA(slow)
            sma_fm: SMA(fast) - SMA(medium)
            sma_ms: SMA(medium) - SMA(slow)
            kama_fm: KAMA(fast) - KAMA(medium)
            kama_ms: KAMA(medium) - KAMA(slow)
            macd: MACD - Signal
            atr: ATR
            adx: ADX
            di: DI+ - DI-
            bb: BB(fast) - BB(medium)
            kc: Keltner(fast) - Keltner(medium)
            stoch: Stochastic - Slow Stochastic
            mom: Momentum
            rsi: RSI
            
        args:
            fast: int: Fast Window
            medium: int: Medium Window
            slow: int: Slow Window
            
        returns:
            DataFrame: containing the states of the indicators
        """
        try:
            adx, diplus, diminus = self.ADX(medium)
            macd, macd_signal = self.macd(fast, medium)
            d = {
                'ema_fm': self.ema(fast) - self.ema(medium),
                'ema_ms': self.ema(medium) - self.ema(slow),
                'sma_fm': self.sma(fast) - self.sma(medium),
                'sma_ms': self.sma(medium) - self.sma(slow),
                'kama_fm': self.KAMA(n=fast, pow1=m, pow2=medium) - self.KAMA(n=medium, pow1=m, pow2=slow),
                'kama_ms': self.KAMA(n=medium, pow1=m, pow2=slow) - self.KAMA(n=slow, pow1=m, pow2=slow * 1.5),
                'macd': (macd - macd_signal).values,
                'atr': self.ATR(fast),
                'adx': adx,
                'di': np.concatenate([[np.nan], diplus]) - np.concatenate([[np.nan], diminus]),
                'bb': self.BB(fast, m) - self.BB(medium, m),
                'kc': self.keltner(fast, m) - self.keltner(medium, m),
                'stoch': self.stochastic(fast)-self.slow_stoch(fast),
                'mom': self.momentum(medium),
                'rsi': self.rsi(medium),
                'z_score': self.mean_reversion(medium),
                'volatility': self.est_vol(lookback=fast)
            }
            return pd.DataFrame(d, index=self.dte_index)
        except Exception as e:
            logger.error(f"Error getting states: {str(e)}")
            raise

if __name__ == "__main__":
    print("Discrimination is a mental skill which allows one to differentiate between what has value and is essential and what is non-essential of no value.")
    import sys 
    sys.path.append("/Users/jerald/Documents/Dir/Python/stocker")
    from bin.main import Manager 
    try:
        M = Manager()
        # Example: Running the indicators on one stock
        prices = M.Pricedb.ohlc('spy')
        G = Indicators(prices)
        techs = G._get_moving_averages()
        print('--'*20, 'SPY', '--'*20)
        print(techs)
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise