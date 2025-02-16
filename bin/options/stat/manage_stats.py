

from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
from tqdm import tqdm
import time 
import logging


import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3])) 

from bin.options.stat.em import Exp
from bin.options.stat.change_vars import ChangeVars
from bin.options.stat.cp import CP

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Stats(Exp, ChangeVars, CP):
    def __init__(self, connections: Dict[str, str]):
        super().__init__(connections)

    def update_stats(self, stock: str, new_chain: pd.DataFrame) -> None:
        """
        Update the Stats Database with the new option data.

        Args:
            stock (str): The ticker symbol of the stock.
            new_chain (pd.DataFrame): New option chain data.

        Raises:
            ValueError: If the input DataFrame is empty.
        """
        if new_chain.empty:
            logger.error(f"Empty DataFrame for stock {stock}")
            raise ValueError("New option chain data cannot be empty")
        self.update_change_vars(stock)
        self.update_cp(stock, new_chain)
        self._em(stock, new_chain)
        self._all_cp()

    def _init_change_db(self) -> None:
        """
        Initialize the change db if needed.

        Raises:
            Exception: If there's an issue initializing the database for any stock.
        """
        for stock in tqdm(self.stocks['all_stocks'], desc="Initializing Change DB"):
            try:
                self._initialize_change_db(stock)
            except Exception as e:
                logger.error(f"Error initializing change db for {stock}: {e}")

    def _init_vol_db(self) -> None:
        """
        Initialize the vol db if needed.

        Raises:
            Exception: If there's an issue initializing the database for any stock.
        """
        for stock in tqdm(self.stocks['all_stocks'], desc="Initializing Vol DB"):
            try:
                self._initialize_vol_db(stock)
            except Exception as e:
                logger.error(f"Error initializing vol db for {stock}: {e}")

    def clear_tables(self) -> None:
        """
        Save a log and clear all tables in the stats db.

        Raises:
            Exception: If there's an error during the backup or table operations.
        """
        try:
            self.stats_db.backup(self.backup)
            cursor = self.stats_db.cursor()
            tables = [i[0] for i in cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]
            for table in tables:
                cursor.execute(f'DROP TABLE {table}')
            self.stats_db.commit()
            logger.info("All tables cleared successfully.")
        except Exception as e:
            logger.error(f"Error occurred while clearing tables: {e}")
            raise

    def cp_query(self, stock: str, n: int = 30) -> pd.DataFrame:
        """
        Fetch and combine old and current CP data.

        Args:
            stock (str): The ticker symbol of the stock.
            n (int): Number of days to look back.

        Returns:
            pd.DataFrame: Combined CP data.

        Raises:
            Exception: If there's an error in reading from the database.
        """
        try:
            old_df = self.get_cp_from_purged_db(stock, n=n)
        except Exception as e:
            logger.warning(f"manage_stats.cp_query: No old CP data for {stock}: {e}")
            old_df = pd.DataFrame()
        
        current_df = self._calculation(self._cp(stock, n=n))
        new_df = pd.concat([old_df, current_df], axis=0).reset_index(drop=True)
        return new_df

    def _init_em_tables(self) -> None:
        """Initialize the expected moves tables."""
        self._initialize_em_tables()

    def reverse_dict(self) -> Dict[str, List[str]]:
        """
        Reverses the keys and values of a dictionary containing string keys and list of string values.

        Returns:
            Dict[str, List[str]]: A new dictionary with reversed keys and values.
        """
        d = self.stocks.copy()
        if 'all_stocks' in d:
            del d['all_stocks']
        reversed_dict = {}
        for key, values in d.items():
            for value in values:
                if value not in reversed_dict:
                    reversed_dict[value] = []
                reversed_dict[value].append(key)
        
        return {stockname: groups[0] if len(groups) == 1 else groups[1] for stockname, groups in reversed_dict.items()}

    def _mdf(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Modify DataFrame by filling in missing data and calculating moving averages.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Modified DataFrame with computed statistics.
        """
        df.total_oi = df.total_oi.ffill()
        df.call_oi = df.call_oi.ffill()
        df.put_oi = df.put_oi.ffill()
        df = df.round(4)

        # Calculate Moving Averages
        n_day = 3
        sma_vol = df['total_vol'].rolling(n_day).mean()
        std_vol = df['total_vol'].rolling(n_day).std()
        sma_oi = df['total_oi'].rolling(n_day).mean()
        std_oi = df['total_oi'].rolling(n_day).std()
        avg_call_change = df['call_oi_chng'].rolling(n_day).mean()
        avg_put_change = df['put_oi_chng'].rolling(n_day).mean()
        pcr_vol = df['put_vol'] / df['call_vol']
        avg_pcr_vol = pcr_vol.rolling(n_day).mean()
        pcr_oi = df['put_oi'] / df['call_oi']
        avg_pcr_oi = pcr_oi.rolling(n_day).mean()
        
        # Insert Moving Averages into DataFrame
        df.insert(3, '30d_avg_oi', sma_oi)
        df.insert(6, '30d_total_oi_std', std_oi)
        df.insert(6, '30d_avg_call_change', avg_call_change)
        df.insert(9, '30d_avg_put_change', avg_put_change)
        df.insert(3, '30d_avg_vol', sma_vol)    
        df.insert(4, '30d_pcr_vol', pcr_vol)
        df.insert(5, '30d_avg_pcr_vol', avg_pcr_vol)
        df.insert(5, '30d_pcr_oi', pcr_oi)
        df.insert(6, '30d_avg_pcr_oi', avg_pcr_oi)
        
        # Clean up DataFrame
        df.dropna(inplace=True)
        for col in ['30d_avg_vol', 'total_vol', '30d_avg_oi', 'total_oi', '30d_avg_call_change', '30d_avg_put_change']:
            df[col] = df[col].astype(int)
        
        return df

    def _all_cp(self) -> pd.DataFrame:
        """
        Returns the Daily Option Stats for all stocks.

        Returns:
            pd.DataFrame: Concatenated DataFrame of daily option stats for all stocks.

        Raises:
            Exception: If data retrieval or processing fails for any stock.
        """
        logger.info("Getting the latest Option stats")
        sg = self.reverse_dict()
        out = []
        for stock in tqdm(self.stocks['all_stocks'], desc="Daily Option Stats"):
            try:
                df = pd.read_sql(f'''SELECT * FROM {stock} ORDER BY datetime(gatherdate) ASC''', self.vol_db, parse_dates=['gatherdate'])
                df['gatherdate'] = pd.to_datetime(df['gatherdate'])
                df.insert(0, 'stock', stock)
                # df.insert(1, 'group', df['stock'].map(sg)) 
                print(df)
                df = self._mdf(df)
                if df.empty:
                    logger.warning(f'manage_stats._all_cp(): No data for {stock}')
                    continue
                out.append(df.tail(1))
            except Exception as e:
                logger.error(f"Error processing stock {stock}: {e}")
        
        combined = pd.concat([x for x in out if not x.empty])
        return combined

if __name__ == "__main__":
    print("Control what you can Control.")
    from bin.main import get_path
    connections = get_path()
    oc = Stats(connections)
    print(oc._all_cp())