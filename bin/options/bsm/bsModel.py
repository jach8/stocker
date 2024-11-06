from scipy.optimize import minimize
import scipy.stats as st 
import numpy as np 
import pandas as pd 
import yfinance as yf 
from warnings import filterwarnings
filterwarnings('ignore')

#print('Downloading Recent 10 year treasury yield data...')
# Start by getting up-to-date risk free rate 
#TEN_YEAR = yf.Ticker('^TNX').history()[['Open','Close', 'High', 'Low']]
#R = TEN_YEAR['Close'].iloc[-1] / 100
R = 0.04
#print(f'10 Yr Treasury Yield: {R:.2f}%')

from scipy.optimize import minimize
import scipy.stats as st 

def get_volatility(market_price, price):
    def error(sigma):
        return (market_price - price)**2
    return minimize(error, 0.2).x[0]

def phi(df):
    df = df.copy()
    df['expiry'] = pd.to_datetime(df['expiry']) 
    df['gatherdate'] = pd.to_datetime(df['gatherdate'])
    days = ((df.expiry+ pd.Timedelta('16:59:59')) - df.gatherdate).dt.days 
    # if days < 1, add 1 to days
    days = days.apply(lambda x: x if x > 0 else x + 1)
    df['timevalue'] = days / 252
    # convert expiry to end at 16:59:59 
    d1 = (np.log(df['stk_price']/df['strike']) + (R + df['impliedvolatility']**2/2)*df['timevalue']) / (df['impliedvolatility'] * np.sqrt(df['timevalue']))
    d2 = d1 - df['impliedvolatility'] * np.sqrt(df['timevalue'])
    nd1 = st.norm.cdf(d1)
    nd2 = st.norm.cdf(d2)
    return d1, d2, nd1, nd2

def call_options(df):
    df = df.copy()
    d1, d2, nd1, nd2 = phi(df)
    discount = np.exp(-R * df['timevalue'])
    df['fairvalue'] = df['stk_price'] * nd1 - df['strike'] * discount * nd2
    df['delta'] = nd1 
    df['gamma'] = st.norm.pdf(d1) / (df['stk_price'] * df['impliedvolatility'] * np.sqrt(df['timevalue']))
    df['theta'] = (df['stk_price']* discount  * st.norm.cdf(d1) * 0  - df['strike'] * discount * R * nd2 - df['stk_price'] * discount * (df['impliedvolatility'] / (2 * np.sqrt(df['timevalue'])) * st.norm.pdf(d1))) / 252
    df['vega'] = (df['stk_price'] * st.norm.pdf(d1) * np.sqrt(df['timevalue'])) * 0.01
    df['rho'] = (df['strike'] * df['timevalue'] * discount * nd2) * 0.01
    return df

def put_options(df):
    df = df.copy()
    discount = np.exp(-R * df['timevalue'])
    d1, d2, nd1, nd2 = phi(df)
    df['fairvalue'] = df['strike'] * discount * st.norm.cdf(-d2) - df['stk_price'] * st.norm.cdf(-d1)
    df['delta'] = st.norm.cdf(-d1) * -discount 
    df['gamma'] = st.norm.pdf(d1) / (df['stk_price'] * df['impliedvolatility'] * np.sqrt(df['timevalue']))
    df['theta'] = (-df['stk_price']* discount  * st.norm.pdf(d1) * 0 + df['strike'] * discount * R * st.norm.cdf(-d2) + df['stk_price'] * discount * (df['impliedvolatility'] / (2 * np.sqrt(df['timevalue'])) * st.norm.pdf(-d1))) / 252
    df['vega'] = (df['stk_price'] * st.norm.pdf(d1) * np.sqrt(df['timevalue'])) * 0.01
    df['rho'] = (-df['strike'] * df['timevalue'] * discount * st.norm.cdf(-d2)) * 0.01
    return df

def bs_df(df):
    df = df.copy()
    calls = df.query('type == "Call"')
    puts = df.query('type == "Put"')
    bsdf = pd.concat([call_options(calls), put_options(puts)])
    if 'index' in bsdf.columns:
        bsdf = bsdf.drop(columns=['index'])
    return bsdf
    


