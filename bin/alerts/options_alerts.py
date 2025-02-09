import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import datetime as dt 
from tqdm import tqdm 
import scipy.stats as st 
import time
import json 
import logging
from typing import List, Optional, Dict
import sqlite3 as sql
from scipy import stats


import sys
sys.path.append('/Users/jerald/Documents/Dir/Python/Stocks')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Notifications:
    def __init__(self, connections: Dict[str, str]):
        try:
            self.vol_db = sql.connect(connections['vol_db'])
            self.stocks = json.load(open(connections['ticker_path'], 'r'))
            logger.info("Notifications instance initialized.")
        except KeyError as e:
            logger.error(f"Key error in connections dict: {e}")
            raise
        except FileNotFoundError:
            logger.error("Ticker file not found.")
            raise

    def stock_data(self, stock: str, n: int = 5, date: Optional[str] = None) -> pd.DataFrame:
        """ 
        Retrieve the daily Option Statistics for a given stock.

        Args:
            stock (str): The stock symbol.
            n (int): Lookback period for change calculations (not used in this method).
            date (Optional[str]): Specific date for data retrieval. If None, use latest data.

        Returns:
            pd.DataFrame: DataFrame with stock option data.

        Raises:
            Exception: If SQL query fails.
        """
        try:
            if date is None:
                df = pd.read_sql(f'select * from {stock}', self.vol_db, parse_dates=['gatherdate'], index_col=['gatherdate'])
            else:
                df = pd.read_sql(f'select * from {stock} where date(gatherdate) <= date("{date}")', self.vol_db, parse_dates=['gatherdate'], index_col=['gatherdate'])
                
            # dropCols = list(df.filter(regex='pct|spread|delta|gamma|theta|vega|prem|iv|total'))
            dropCols = list(df.filter(regex='pct|spread|total|atm|otm|iv_chng'))
            df = df.drop(columns=dropCols)
            return df
        except Exception as e:
            logger.error(f"Error retrieving stock data for {stock}: {e}")
            return pd.DataFrame()

    def col_map(self, col: str) -> str:
        """ 
        Map column names to human-readable descriptions.

        Args:
            col (str): Column name to map.

        Returns:
            str: Mapped description.

        Raises:
            KeyError: If column name is not in the mapping dictionary.
        """
        cmap = {
                    "call_vol": "Call Volume",
                    "put_vol": "Put Volume",
                    "total_vol": "Total Volume",
                    "call_oi": "Call OI",
                    "put_oi": "Put OI",
                    "total_oi": "Total OI",
                    "call_prem": "Call Premium",
                    "put_prem": "Put Premium",
                    "total_prem": "Total Premium",
                    "call_iv": "Call IV",
                    "put_iv": "Put IV",
                    "atm_iv": "ATM IV",
                    "otm_iv": "OTM IV",
                    "call_vol_chng": "Call Volume Chng",
                    "put_vol_chng": "Put Volume Chng",
                    "total_vol_chng": "Total Volume Chng",
                    "call_oi_chng": "Call OI Chng",
                    "put_oi_chng": "Put OI Chng",
                    "total_oi_chng": "Total OI Chng",
                    "call_prem_chng": "Call Prem. Chng",
                    "put_prem_chng": "Put Prem. Chng",
                    "total_prem_chng": "Total Prem. Chng",
                    "call_iv_chng": "Call IV Chng",
                    "put_iv_chng": "Put IV Chng",
                    "atm_iv_chng": "ATM IV Chng",
                    "otm_iv_chng": "OTM IV Chng",
                    "call_oi_chng5d": "Call OI Chng (5d)",
                    "put_oi_chng5d": "Put OI Chng (5d)",
                    "call_vol_chng5d": "Call Vol Chng (5d)",
                    "put_vol_chng5d": "Put Vol Chng (5d)",
                }
        try:
            return cmap[col]
        except KeyError:
            logger.warning(f"Column {col} not found in col_map")
            return col

    def _colors(self, color: Optional[str] = None, word: Optional[str] = None) -> str:
        """ 
        Generate ANSI color codes or apply color to text.

        Args:
            color (Optional[str]): Name of color or None for all colors.
            word (Optional[str]): Text to colorize.

        Returns:
            str: Color code or colored text.

        Raises:
            KeyError: If an invalid color name is provided.
        """
        colors = {
            'header': '\033[95m', 'red': '\033[31m', 'green': '\033[32m', 'yellow': '\033[33m',
            'blue': '\033[34m', 'cyan': '\033[36m', 'white': '\033[37m', 'grey': '\033[30m',
            'bright-red': '\033[91m', 'bright-green': '\033[92m', 'bright-yellow': '\033[93m',
            'end': '\033[0m',
        }
        
        if color is None:
            return colors
        elif word is None:
            return colors.get(color, '')
        else:
            return colors.get(color, '') + word + colors['end']
        
    ##############################################  Metrics  ########################################################
    
    
    def _percentile_score(self, df: pd.DataFrame, col: str) -> float:
        """ 
        Calculate percentile score for a column in a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            col (str): The column name to analyze.

        Returns:
            float: Percentile score where 0 indicates unusually high values.

        Raises:
            ValueError: If column does not exist in DataFrame.
        """
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
        return st.percentileofscore(df[col], df[col].iloc[-1], kind='strict', nan_policy='omit')
    
    def calculate_ivr(self, df, col):
        """
        Calculate Implied Volatility Rank (IVR).

        Args:
            df (pd.DataFrame): DataFrame containing historical data.
            col (str): Column name for IV data.

        Returns:
            float: IVR between 0 and 100.
        """

        iv = df[col].iloc[-1]
        iv_52w_high = df[col].rolling(window=252, min_periods=1).max().iloc[-1]
        iv_52w_low = df[col].rolling(window=252,  min_periods=1).min().iloc[-1]
        
        if iv_52w_high == iv_52w_low:
            return None
        
        ivr = (iv - iv_52w_low) / (iv_52w_high - iv_52w_low) * 100
        if ivr is np.nan:
            return None
        else:
            return ivr

    def calculate_ivp(self, df, col):
        """
        Calculate Implied Volatility Percentile (IVP).

        Args:
            df (pd.DataFrame): DataFrame containing historical data.
            col (str): Column name for IV data.

        Returns:
            float: IVP between 0 and 100.
        """
        current_iv = df[col].iloc[-1]
        historical_iv = df[col].rolling(window=252, min_periods=10).mean()
        percentile = stats.percentileofscore(historical_iv, current_iv)
        return percentile

    def open_interest_change_rate(self, df, col, period=1):
        """
        Calculate the rate of change in open interest.

        Args:
            df (pd.DataFrame): DataFrame with open interest data.
            col (str): Column name for OI.
            period (int): Number of periods to look back.

        Returns:
            float: Change rate as a percentage.
        """
        current_oi = df[col].iloc[-1]
        previous_oi = df[col].iloc[-period]
        if previous_oi == 0:
            return 0  # Avoid division by zero
        change_rate = (current_oi - previous_oi) / previous_oi * 100
        return change_rate

    def z_score(self, df, col):
        """
        Calculate the Z-score for a given column.

        Args:
            df (pd.DataFrame): DataFrame with the data.
            col (str): Column name.

        Returns:
            float: Z-score of the last entry in the column.
        """
        last_value = df[col].iloc[-1]
        mean = df[col].mean()
        std_dev = df[col].std()
        if std_dev == 0:
            return 0  # Avoid division by zero
        return (last_value - mean) / std_dev        

    

    ##############################################  NOTIFICATIONS  ########################################################
    def __generate_text(self, stock: str, df: pd.DataFrame, col: str) -> Optional[str]:
        """
        Generate text notifications based on changes in option metrics.

        Args:
            stock (str): Stock symbol.
            df (pd.DataFrame): DataFrame with stock data.
            col (str): Column to analyze.

        Returns:
            Optional[str]: Notification text or None if not applicable.
        """
        if col not in df.columns:
            logger.warning(f"Column {col} not found in DataFrame for {stock}")
            return None

        try:
            current = df[col].iloc[-1]
            previous = df[col].iloc[-2] if df.shape[0] > 1 else None
            ma_20 = df[col].rolling(window=20).mean().iloc[-1] if df.shape[0] > 20 else None
            historical_max = df[col].max()
            historical_min = df[col].min()
            
            # Color coding for 'Call' and 'Put'
            # col_name = self.col_map(col)
            # if 'Call' in col_name:
            #     col_name = col_name.replace('Call', self._colors('green', 'Call'))
            # elif 'Put' in col_name:
            #     col_name = col_name.replace('Put', self._colors('red', 'Put'))
            col_name = self.col_map(col)
            
            run_ivr = False
            # Added new metrics
            if 'iv' in col:
                try:
                    ivr = self.calculate_ivr(df, col)
                    ivp = self.calculate_ivp(df, col)
                    run_ivr = True
                except Exception as e:
                    logger.error(f"Error calculating IVR or IVP for {stock}, column {col}: {e}, {ivr}, {ivp}")
                    ivr, ivp = None, None
                
                
            oi_change_rate = self.open_interest_change_rate(df, col) if 'oi' in col else None
            z_score_val = self.z_score(df, col)

            # Percentage change from last day
            if previous is not None:
                pct_change = (current - previous) / previous * 100
                if abs(pct_change) > 100:  
                    color = 'green' if pct_change > 0 else 'red'
                    txt = f"${stock.upper()} {col_name} changed by {pct_change:.2f}% from {previous:.2f} to {current:,.2f}"
                    txt = self._colors(color, txt)
                    logger.info(f"{txt}")
                    return txt
            
            # At 20-day moving average
            if ma_20 is not None:
                deviation = (current - ma_20) / ma_20 * 100
                if abs(deviation) < .01:
                    txt = f"${stock.upper()} {col_name} is {deviation:.2f}% at its 20-day moving average"
                    txt = self._colors('grey', txt)
                    logger.info(f"{txt}")
                    return txt

            # IVR and IVP notifications (for IV columns)
            if run_ivr == True:
                if ivr > 80:
                    txt = f"${stock.upper()} {col_name} IVR is at {ivr:.2f}%, suggesting options are expensive"
                    txt = self._colors('yellow', txt)
                    logger.info(f"{txt}")
                    return txt
                elif ivr < 20:
                    txt = f"${stock.upper()} {col_name} IVR is at {ivr:.2f}%, suggesting options are cheap"
                    txt = self._colors('bright-yellow', txt)
                    logger.info(f"{txt}")
                    return txt

                if ivp > 90:
                    txt = f"${stock.upper()} {col_name} is in the {ivp:.2f}th percentile of historical volatility"
                    txt = self._colors('yellow', txt)
                    logger.info(f"{txt}")
                    return txt
                elif ivp < 10:
                    txt = f"${stock.upper()} {col_name} is in the {ivp:.2f}th percentile, indicating low volatility"
                    txt = self._colors('yellow', txt)
                    logger.info(f"{txt}")
                    return txt

            # Open Interest Change Rate
            if oi_change_rate is not None:
                if abs(oi_change_rate) > 5:  # Arbitrary threshold, adjust as needed
                    direction = "increased" if oi_change_rate > 0 else "decreased"
                    txt = f"${stock.upper()} {col_name} has {direction} by {abs(oi_change_rate):.2f}%"
                    txt = self._colors('green' if oi_change_rate > 0 else 'red', txt)
                    logger.info(f"{txt}")
                    return txt

            # Z-Score
            if abs(z_score_val) > 2:  # Typically, values > 2 or < -2 are considered significant
                txt = f"${stock.upper()} {col_name} has a z-score of {z_score_val:.2f}, indicating an outlier"
                txt = self._colors('teal', txt)
                logger.info(f"{txt}")
                return txt

            col_name = self.col_map(col)
            
            # Percentage change from last day
            if previous is not None:
                pct_change = (current - previous) / previous * 100
                if abs(pct_change) > 1000:  # Notify if change > 10%
                    color = 'green' if pct_change > 2 else 'red'
                    txt = f"${stock.upper()} {col_name} changed by {pct_change:.2f}% from {previous:.2f} to {current:,.2f}"
                    txt = self._colors(color, txt)
                    logger.info(f"{txt}")
                    return txt
            
            # Deviation from 20-day moving average
            if ma_20 is not None:
                deviation = (current - ma_20) / ma_20 * 100
                if abs(deviation) < 0.5:  # Notify if deviation > 20%
                    txt = f"${stock.upper()} {col_name} is {deviation:.2f}% within 1% of the 20-day moving average"
                    txt = self._colors('blue', txt)
                    logger.info(f"{txt}")
                    return txt
            
            # Check for high volume or open interest changes (for vol or oi columns)
            if 'vol' in col or 'oi' in col:
                if df.shape[0] > 5:
                    five_day_avg = df[col].iloc[-5:].mean()
                    if current > five_day_avg * 5:  # Notify if current is double the 5-day average
                        txt = f"${stock.upper()} {col_name} surged to {current:,.2f}, 3x higher than 5-day average"
                        txt = self._colors('bright-yellow', txt)
                        logger.info(f"{txt}")
                        return txt

            # Check for all-time highs or lows, or close to them
            if current >= historical_max or (current >= historical_max * 0.99):  # Within 1% of all-time high
                txt = f"${stock.upper()} {col_name} at new all-time high or within 1% of it: {current:,.2f}"
                txt = self._colors('bright-green', txt)
                logger.info(f"{txt}")
                return txt
            
            if current <= historical_min or (current <= historical_min * 1.01):  # Within 1% of all-time low
                txt = f"${stock.upper()} {col_name} plummeted to a new all-time low or within 1% of it: {current:,.2f}"
                txt = self._colors('bright-red', txt)
                logger.info(f"{txt}")
                return txt

        except IndexError:
            logger.error(f"Index error in column {col} for {stock}")
        except Exception as e:
            logger.error(f"Error in __generate_text for {stock}, column {col}: {e}")

        return None
    

    def __remove_colors(self, txt: str) -> str:
        """ 
        Remove ANSI color codes from text.

        Args:
            txt (str): Text containing color codes.

        Returns:
            str: Text without color codes.
        """
        colors = [v for v in self._colors().values()]
        for color in colors:
            txt = txt.replace(color, '')
        return txt

    def notifications(self, stock: str, n: int = 5, date: Optional[str] = None) -> List[str]:
        """ 
        Generate notifications for a specific stock.

        Args:
            stock (str): Stock symbol.
            n (int): Lookback period.
            date (Optional[str]): Specific date for data.

        Returns:
            List[str]: List of notification texts.

        Raises:
            ValueError: If stock is not in the known stocks list.
        """
        if stock not in self.stocks['all_stocks']:
            logger.warning(f"Stock {stock} not found in known stocks.")
            return []
        
        df = self.stock_data(stock, n, date)
        if df.empty or df.shape[0] <= 10:
            return []
        
        for i in df.columns:
            txt = self.__generate_text(stock, df, i)
            if txt is not None:
                # logger.info(f"Notification for {stock}: {txt}")
                pass



    def iterator(self, n: int = 5, date: Optional[str] = None) -> List[str]:
        """ 
        Iterate through all stocks to generate notifications.

        Args:
            n (int): Number of days for data analysis.
            date (Optional[str]): Specific date for analysis.

        Returns:
            List[str]: All generated notifications.

        Raises:
            Exception: If there's an issue iterating over stocks.
        """
        out = []
        for stock in tqdm(self.stocks['all_stocks'], desc="Generating Notifications"):
            try:
                notifications = self.notifications(stock, n, date)
                if notifications:
                    out.extend(notifications)
                # time.sleep(np.random.normal(3, 0.2))  # Consider if this can be optimized or removed
            except Exception as e:
                logger.error(f"Error processing notifications for {stock}: {e}")
        return out

if __name__ == '__main__':
    from bin.main import get_path
    connections = get_path()
    notif = Notifications(connections)
    try:
        out = notif.iterator(n=3, date=None)
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
