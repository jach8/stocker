import pickle
import pandas as pd 
import sqlite3 
import os 
import json 
import datetime as dt 
import numpy as np 

import sys
sys.path.append('/Users/jerald/Documents/Dir/Python/stocker')

from bin.price.db_connect import Prices
from bin.options.manage_all import Manager as Optionsdb
from bin.earnings.get_earnings import Earnings 
from bin.price.perf.report import perf as performance
from bin.alerts.options_alerts import Notifications
from bin.signals.plays.ling import Scanner
from bin.signals.plays.dxp import dxp 
from bin.utils.Initialize import Initialize
from bin.utils.add_stocks import add_stock

def init():
    """ Initialize the program. """
    Initialize()
    
    
def get_path(pre=''):
	""" Must be in the '/Users/jerald/Documents/Dir/Python/stocker' directory to run this function. """
	connections = {
				##### Bonds Data ###########################
				'bonds_db': f'{pre}data/bonds/bonds.db', 
				##### Price Data ###########################
				'daily_db': f'{pre}data/prices/stocks.db', 
				'intraday_db': f'{pre}data/prices/stocks_intraday.db',
				'ticker_path': f'{pre}data/stocks/tickers.json',
				##### Options Data ###########################
				'inactive_db': f'{pre}data/options/log/inactive.db',
				'backup_db': f'{pre}data/options/log/backup.db',
				'tracking_values_db': f'{pre}data/options/tracking_values.db',
				'tracking_db': f'{pre}data/options/tracking.db',
				'stats_db': f'{pre}data/options/stats.db',
				'vol_db': f'{pre}data/options/vol.db',
				'change_db': f'{pre}data/options/option_change.db', 
				'option_db': f'{pre}data/options/options.db', 
				##### Earnings + Company Info ###########################
				'earnings_dict': f'{pre}data/earnings/earnings.pkl',
				'stock_names' : f'{pre}data/stocks/stock_names.db',
                'stock_info_dict': f'{pre}data/stocks/stock_info.json',
                # 'earnings_calendar': f'{pre}data/earnings/earnings_dates_alpha.csv',
		}
	return connections

def check_path(connections):
	# Check if the files exist
	checks = []
	for key, value in connections.items():
		if not os.path.exists(value):
			raise ValueError(f'{value} does not exist')
		checks.append(True)
	return all(checks)
           	 
class Manager:
	def __init__(self, connections=None):
		#  If type is string, or is None
		if type(connections) == str:
			connections = get_path(connections)
		if connections == None:
			connections = get_path()
		if type(connections) != dict:
			raise ValueError('Connections must be a dictionary')
		# Check if the files exist
		if check_path(connections) == False:
			raise ValueError('Files do not exist')

		# Initialize the Connections: 
		self.Optionsdb = Optionsdb(connections) 
		self.Pricedb = Prices(connections)
		# self.Earningsdb = Earnings(connections)
		self.performance = performance(connections) 
		self.Notifications = Notifications(connections)
		self.Scanner = Scanner(connections)
		self.dxp = dxp(connections) 
		
		# Save the Connection Dict.
		self.connection_paths = connections
		  
	def close_connection(self):
		self.Optionsdb.close_connections()
		self.Pricedb.close_connections()
  
	def addStock(self, stock):
		conn = sqlite3.connect(self.connection_paths['stock_names'])
		add_stock(conn, self.connection_paths['ticker_path'], stock)

	
if __name__ == "__main__":
	Initialize()
 
 