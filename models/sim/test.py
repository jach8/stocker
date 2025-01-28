import numpy as np 
import re
import datetime as dt 
import matplotlib.pyplot as plt 
import pandas as pd
import sys
from tqdm import tqdm   
from models.sim.lsm import OptionSim as sim

def prep_df(df, rf_rate = 0.45, nsims = 100):
    """ 
    Prep the DataFrame of Options data for the LSM Model
        
        Args: 
            df: pd.DataFrame: The DataFrame of Options Data
            rf_rate: float: The Risk Free Rate
            nsims: int: The Number of Simulations to Run 
    
    
    """
    df['rf']  = rf_rate
    df['nsim'] = nsims
    df['impliedvolatility'] = df.impliedvolatility.astype(float)
    df['strike'] = df.strike.astype(float)
    df['stk_price'] = df.stk_price.astype(float)
    df['lastprice'] = df.lastprice.astype(float)
    df.expiry = pd.to_datetime(df.expiry)
    df.gatherdate = pd.to_datetime(df.gatherdate)
    df['dte'] = ((df.expiry - df.gatherdate).dt.days).astype(float) 
    # df['timevalue'] = df.timevalue.astype(float) / 1e7
    df['timevalue'] = df.dte / 252
    return df


def lsm(df, rf_rate = 0.0379, nsims = 1000, jump = False):
    df = prep_df(df, rf_rate, nsims) 
    lodf = []
    vals = list(zip(df.stk_price, df.strike, df.rf, df.dte,df.timevalue, df.impliedvolatility,df.type, df.nsim ,df.lastprice))
    for s, k, r, dte, tv,  sig, ty, nsim, lp in tqdm(vals):
        s = sim(s, k, r, dte, tv, sig, ty, nsim, lp)
        lodf.append(s.run(jump = jump))
    
    return pd.concat(lodf)


if __name__ == '__main__':
    from bin.main import Manager 
    
    m = Manager()
    stat_df = pd.read_sql('select * from oi_chg order by volume desc ', m.Optionsdb.stats_db)
    print(stat_df.shape)
    # df = m.Optionsdb.parse_change_db(stat_df)
    # df['type'] = df['type'].apply(lambda x: 'Call' if x == 'C' else 'Put')
    df = m.Optionsdb.pcdb(stat_df)
    df = m.Optionsdb.new_bsdf_df(df)
    df = prep_df(df)
    ndf = lsm(df, jump = True)
    ndf.insert(0, 'contractsymbol', df.contractsymbol)
    df = df.merge(ndf, on = 'contractsymbol')
    print(df)
