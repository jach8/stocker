import json
import pandas as pd 
import numpy as np 
import sqlite3 as sql
import datetime as dt
from tqdm import tqdm 
import time
from bin.main import Manager, get_path, Initialize
from bin.alerts.options_alerts import Notifications


class Pipeline(Manager):
    def __init__(self, connections = None):
        super().__init__(connections)

    def update_options(self):
        out = []
        begin = '\033[92m'
        endoc = '\033[0m'
        pbar = tqdm(self.Optionsdb.stocks['all_stocks'], desc = "Options Data")
        blips = []
        for stock in pbar:
            pbar.set_description(f"Options Data {begin}${stock.upper()}{endoc}")
            new_chain = self.Optionsdb.insert_new_chain(stock)
            if new_chain is None:
                continue
            else:
                # self._update_change_vars(stock)
                self.Optionsdb._initialize_change_db(stock)
                # self.update_cp(stock, new_chain)
                self.Optionsdb._intialized_cp(stock)
                # self.run_screener(stock, new_chain)
                self.Notifications.notifications(stock, n = 10)
                time.sleep(2.5)   
        
        # Gather Todays Option Stats 
        self.Optionsdb._all_cp()
        
        # Gather the Expected Move Data 
        self.Optionsdb._init_em_tables()
        
        # Scan for the plays
        self.Scanner.scan_contracts()
        
    def workflow(self):
        # Gather Todays Option Stats 
        self.Optionsdb._all_cp()
        
        # Gather the Expected Move Data 
        self.Optionsdb._init_em_tables()
        
        # Scan for the plays
        self.Scanner.scan_contracts()
        
    
        
    def update_stock_prices(self):
        self.Pricedb.update_stock_prices()
        self.performance.show_performance()
    
    def view_notifications(self):
        self.Notifications.iterator()
        
    def master_run(self):
        # Update the stock prices
        self.update_stock_prices()
        # Print the price performance
        self.performance.show_performance()
        # Update the options data 
        self.update_options() 
        self.workflow()


if __name__ == "__main__":
    print("\n12.8 Concentrate the mind upon Me, apply spiritual intelligence for Me; verily you will reside with me after this existence without a doubt.\n")
    Initialize()
    m = Pipeline()
    # m.master_run()
    # m.close_connection()
    m.master_run()