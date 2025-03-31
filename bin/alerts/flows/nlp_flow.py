from gflow import cp_data_utility
import pandas as pd 
import numpy as np 
import json
import sqlite3 as sql
import datetime as dt
from itertools import chain
from typing import Dict, List, Tuple, Union




class cp_nlp_utility:
    def __init__(self, connections: Dict[str, Union[str, sql.Connection]]):
        """
        Initialize the cp_nlp_utility class.
        
        Args:
            connections (Dict[str, Union[str, sql.Connection]]): Dictionary containing database and file paths
        """

        self.data_utility = cp_data_utility(connections)
        self.tickers = self.data_utility.stock_dict['all_stocks']


    def workflow(self): 
        self.data_utility.get_stocks()


if __name__ == "__main__":
    from pathlib import Path 
    import sys 
    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from bin.main import get_path 

    connections = get_path()


    nlp = cp_nlp_utility(connections)

    nlp.workflow()
