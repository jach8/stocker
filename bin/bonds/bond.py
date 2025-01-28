import sqlite3 as sql
import pandas as pd
import datetime as dt
# import nasdaqdatalink
import os 

import sys
# from bin.bonds.qkeys.card import nasdaq as nd 

class bonds():
    def __init__(self, connections):
        self.bonds_db = sql.connect(connections['bonds_db'])
        
    def update_bonds(self):
        os.environ["NASDAQ_DATA_LINK_API_KEY"] = nd
        us_bonds = nasdaqdatalink.get('USTREASURY/YIELD')
        print(us_bonds)
        us_bonds.columns = [x.replace(" ", "") for x in us_bonds.columns]
        us_bonds.to_sql('us_bonds', self.bonds_db, if_exists = 'replace')
        print('Bonds Updated')
    
    def bond_df(self):
        return pd.read_sql('select * from us_bonds', self.bonds_db, parse_dates = ['Date'], index_col = ['Date']).sort_index()
    
if __name__ == '__main__':
    from bin.main import get_path 
    
    c = get_path()
    b = bonds(c)
    b.update_bonds()
    # print(b.bond_df())
    