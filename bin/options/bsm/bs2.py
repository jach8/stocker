"""
Black Scholes Model for Pricing Options and Calculating the Greeks: 
    This model is used to calculate the price of an option given the following inputs:
        - Stock Price
        - Strike Price
        - Time to Expiry
        - Risk Free Rate
        - Implied Volatility
        


    Greeks: 
        - delta (change in the option price with respect to the change in the price of the underlying asset)
        - gamma (change in delta with respect to the underlying price) 
        - theta (change in the option price with respect to time)
        - vega (change in the option price with respect to IV)
        - rho (change in option price with respect to the risk free rate)
        - lanmbda (change in the option price per percentage change in the price of the underlying asset)
        
        *** Second Order Greeks ( Still need to validate for correctness) ***
        
        - vanna (change in delta with respect to IV)
        - charm (change in delta with respect to time)
        - volga (change in vega with respect to IV)
        - veta (change in vega with respect to time)
        - speed (change in gamma with respect to the underlying price)
        - zomma (change in gamma with respect to IV)
        - color (change in gamma with respect to time)
        - ultima (change in vega with respect to the underlying price)
    

"""


from scipy.optimize import minimize, minimize_scalar
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


def get_timeValue(expiry, gatherdate):
    """ Return the time to expiration in minutes """
    hr, minute = 17, 30 
    expiry = pd.to_datetime(expiry)
    # add the hour and minute to the expiry date 
    exp_dt = expiry + pd.Timedelta(hours=hr, minutes=minute)   
    
    gatherdate = pd.to_datetime(gatherdate)
    cur_hr, cur_minute = gatherdate.hour, gatherdate.minute
    
    days = 24 * 60 * 60 *(exp_dt - gatherdate).days
    seconds = (exp_dt - gatherdate).seconds
    return days + seconds / (252 * 24 * 60 * 60)

def time_value(df):
    df['timevalue'] = df.apply(lambda x: get_timeValue(x.expiry, x.gatherdate), axis =1 )
    return df

def phi(df):
    df = df.copy()
    df['expiry'] = pd.to_datetime(df['expiry']) 
    df['gatherdate'] = pd.to_datetime(df['gatherdate'])
    if 'timevalue' not in list(df.columns):
        df['timevalue'] = df.apply(lambda x: get_timeValue(x.expiry, x.gatherdate), axis =1 )
        
    d1 = (np.log(df['stk_price']/df['strike']) + (R + df['impliedvolatility']**2/2)*df['timevalue']) / (df['impliedvolatility'] * np.sqrt(df['timevalue']))
    d2 = d1 - df['impliedvolatility'] * np.sqrt(df['timevalue'])
    nd1 = st.norm.cdf(d1)
    nd2 = st.norm.cdf(d2)
    return d1, d2, nd1, nd2

def call_options(df):
    df = df.copy()
    d1, d2, nd1, nd2 = phi(df)
    discount = np.exp(-R * df['timevalue'])
    pv = df['strike'] * discount 
    df['fairvalue'] = df['stk_price'] * nd1 - df['strike'] * discount * nd2
    df['delta'] = nd1 
    df['gamma'] = st.norm.pdf(d1) / (df['stk_price'] * df['impliedvolatility'] * np.sqrt(df['timevalue']))
    
    # Vanna per 1% change in IV [0.01 * -e ** (-self.q * self.T) * self.d2 / self.sigma * norm.pdf(self.d1)]
    df['vanna'] = 0.01 * -discount * d2 / df['impliedvolatility'] * st.norm.pdf(d1)
    # df['vanna'] = np.sqrt(df['timevalue']) * st.norm.pdf(d1) * (d2 / df['impliedvolatility'])
    
    # df['charm'] = (R * discount - st.norm.pdf(d1) * df['impliedvolatility'] / (2 * np.sqrt(df['timevalue']))) * -1 * nd1
    df['charm'] = ((1/252) * (-discount) * st.norm.pdf(d1) * R)+ st.norm.cdf(d1)
    
    # Color
    #  mess = ((2*(r - q)*t - d2*v*m.sqrt(t)) / (v*m.sqrt(t)))*d1
    #  (-pv*(phid1 / (2*S*t*v*m.sqrt(t)))           *(2*q*t + 1 + mess)) / 252
    inner = ((2 * (R) * df['timevalue']) / (df['impliedvolatility'] * np.sqrt(df['timevalue']) * d1))
    df['color'] = (- pv * (st.norm.pdf(d1) / (2 * df['stk_price'] * df['timevalue'] * df['impliedvolatility'] * np.sqrt(df['timevalue'])))) * (2 * df['timevalue'] + 1 + inner) / 252
    
    # df['vega'] = (df['stk_price'] * st.norm.pdf(d1) * np.sqrt(df['timevalue'])) * 0.01
    df['vega'] = (df['strike'] * discount) *  st.norm.pdf(d2) * np.sqrt(df['timevalue']) * 0.01
    df['volga'] = df['vega'] * (d1 * d2) / df['impliedvolatility']
    df['theta'] = (df['stk_price']* discount  * st.norm.cdf(d1) * 0  - df['strike'] * discount * R * nd2 - df['stk_price'] * discount * (df['impliedvolatility'] / (2 * np.sqrt(df['timevalue'])) * st.norm.pdf(d1))) / 252
    df['rho'] = (df['strike'] * df['timevalue'] * discount * nd2) * 0.01
    df['lam'] = df['delta'] * (df['stk_price'] / df['lastprice'])
    df['speed'] = - (df['gamma'] / df['stk_price']) * (d1 / (df['impliedvolatility'] * np.sqrt(df['timevalue']) + 1))
    df['zomma'] = df['gamma']  * ((d1 * d2 - 1) / df['impliedvolatility']) # df['dexp'] = 100 * df.openinterest * df.delta * df.stk_price
    df['dexp'] = df.delta * df.openinterest * df.stk_price
    df['gexp'] = df.openinterest * df.gamma * df.stk_price**2 
    df['vexp'] = df.vanna * df.openinterest * df.stk_price * df.impliedvolatility
    # df['vexp'] = 100 * df.openinterest * df.vanna * df.stk_price  
    df['cexp'] = df.charm * df.openinterest * df.stk_price * df.timevalue
    return df

def put_options(df):
    df = df.copy()
    discount = np.exp(-R * df['timevalue'])
    pv = df['strike'] * discount
    d1, d2, nd1, nd2 = phi(df)
    df['fairvalue'] = df['strike'] * discount * st.norm.cdf(-d2) - df['stk_price'] * st.norm.cdf(-d1)
    df['delta'] = st.norm.cdf(-d1) * -discount 
    df['gamma'] = st.norm.pdf(d1) / (df['stk_price'] * df['impliedvolatility'] * np.sqrt(df['timevalue']))
    # df['vanna'] = np.sqrt(df['timevalue']) * st.norm.pdf(d1) * (d2 / df['impliedvolatility']) 
    
    # Vanna per 1% change in IV [0.01 * -e ** (-self.q * self.T) * self.d2 / self.sigma * norm.pdf(self.d1)]
    df['vanna'] = 0.01 * -discount * d2 / df['impliedvolatility'] * st.norm.pdf(d1)
    # df['vanna'] = np.sqrt(df['timevalue']) * st.norm.pdf(d1) * (d2 / df['impliedvolatility'])
    
    # df['charm'] = (R * discount - st.norm.pdf(d1) * df['impliedvolatility'] / (2 * np.sqrt(df['timevalue']))) * -1 * nd1
    df['charm'] = ((1/252) * (-discount) * st.norm.pdf(d1) * R)+ st.norm.cdf(-d1)
    
    # Color
    #  mess = ((2*(r - q)*t - d2*v*m.sqrt(t)) / (v*m.sqrt(t)))*d1
    #  (-pv*(phid1 / (2*S*t*v*m.sqrt(t)))           *(2*q*t + 1 + mess)) / 252
    inner = ((2 * (R) * df['timevalue']) / (df['impliedvolatility'] * np.sqrt(df['timevalue']) * d1))
    df['color'] = (- pv * (st.norm.pdf(d1) / (2 * df['stk_price'] * df['timevalue'] * df['impliedvolatility'] * np.sqrt(df['timevalue'])))) * (2 * df['timevalue'] + 1 + inner) / 252
    
    # df['vega'] = (df['stk_price'] * st.norm.pdf(d1) * np.sqrt(df['timevalue'])) * 0.01
    df['vega'] = (df['strike'] * discount) *  st.norm.pdf(d2) * np.sqrt(df['timevalue']) * 0.01
    df['volga'] = df['vega'] * (d1 * d2) / df['impliedvolatility']
    df['theta'] = (  df['stk_price']* discount  * st.norm.pdf(d1) * 0 + df['strike'] * discount * R * st.norm.cdf(-d2) + df['stk_price'] * discount * (df['impliedvolatility'] / (2 * np.sqrt(df['timevalue'])) * st.norm.pdf(-d1))) / 252
    df['rho'] = (-df['strike'] * df['timevalue'] * discount * st.norm.cdf(-d2)) * 0.01
    df['lam'] = df['delta'] * (df['stk_price'] / df['lastprice'])
    df['speed'] = - (df['gamma'] / df['stk_price']) * (d1 / (df['impliedvolatility'] * np.sqrt(df['timevalue']) + 1))
    df['zomma'] = df['gamma']  * ((d1 * d2 - 1) / df['impliedvolatility'])
    # df['dexp'] = -100 * df.openinterest * df.delta * df.stk_price
    df['dexp'] = - df.delta * df.openinterest * df.stk_price
    df['gexp'] = - df.openinterest * df.gamma * df.stk_price**2 
    df['vexp'] = df.vanna * df.openinterest * df.stk_price * df.impliedvolatility
    # df['vexp'] = -100 * df.openinterest * df.vanna * df.stk_price 
    df['cexp'] = - df.charm * df.openinterest * df.stk_price * df.timevalue
    return df

def bs_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * st.norm.cdf(d1) - K * np.exp(-r * T) * st.norm.cdf(d2)

def bs_put(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * st.norm.cdf(-d2) - S * st.norm.cdf(-d1)

def imp_vol(option_value, S, K, T, r, type):
    def call_obj(sigma):
        return np.abs(bs_call(S, K, T, r, sigma) - option_value)
    def put_obj(sigma):
        return np.abs(bs_put(S, K, T, r, sigma) - option_value)
    if type == 'Call':
        return minimize_scalar(call_obj).x
    else:
        return minimize_scalar(put_obj).x
    
def iv(row):
    S = row['stk_price']
    K = row['strike']
    T = row['timevalue']
    r = R
    option_value = row['lastprice']
    if row['type'] == 'Call':
        def call_obj(sigma):
            return np.abs(bs_call(S, K, T, r, sigma) - option_value)
        return minimize_scalar(call_obj).x
    else:
        def put_obj(sigma):
            return np.abs(bs_put(S, K, T, r, sigma) - option_value)
        return minimize_scalar(put_obj).x
    
def bs_df(df):
    df = df.copy()
    if 'timevalue' not in list(df.columns):
        # df['timevalue'] = df.apply(lambda x: get_timeValue(x.expiry, x.gatherdate), axis =1 )
        df['timevalue'] = (pd.to_datetime(df['expiry']) - pd.to_datetime(df['gatherdate'])).dt.days / 252
        df['timevalue'] = np.abs(df['timevalue'])
    df.openinterest = np.where(df.openinterest == 0, .1e-9, df.openinterest)
    df.timevalue = np.abs(df.timevalue)
    calls = df.query('type == "Call"')
    puts = df.query('type == "Put"')
    bsdf = pd.concat([call_options(calls), put_options(puts)])
    # bsdf['iv_fit'] = bsdf.apply(iv, axis = 1)
    if 'index' in bsdf.columns:
        bsdf = bsdf.drop(columns=['index'])
    return bsdf
    

if __name__ == "__main__":
    print("\n\n I often wonder if I made the right choices in life. \n\n")
    
    import sqlite3 as sql 
    import sys
    from pathlib import Path    
    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from bin.options.manage_all import Manager 
    
    
    pre = ''
    connections = {
                ##### Price Report ###########################
                'daily_db': f'{pre}data/prices/stocks.db', 
                'intraday_db': f'{pre}data/prices/stocks_intraday.db',
                'ticker_path': f'{pre}data/stocks/tickers.json',
                ##### Price Report ###########################
                'inactive_db': f'{pre}data/options/log/inactive.db',
                'backup_db': f'{pre}data/options/log/backup.db',
                'tracking_values_db': f'{pre}data/options/tracking_values.db',
                'tracking_db': f'{pre}data/options/tracking.db',
                'stats_db': f'{pre}data/options/stats.db',
                'vol_db': f'{pre}data/options/vol.db',
                'change_db': f'{pre}data/options/option_change.db', 
                'option_db': f'{pre}data/options/options.db', 
                'options_stat': f'{pre}data/options/options_stat.db',
                'stock_names' : f'{pre}data/stocks/stocks.db'
    }
    m = Manager(connections)
    q = 'select * from dia where date(gatherdate) = (select max(date(gatherdate)) from dia)'
    # df = pd.read_sql(q, m.option_db)    
    cursor = m.option_db.cursor()
    ql = cursor.execute(q)
    df = pd.DataFrame(ql.fetchall(), columns = [desc[0] for desc in cursor.description])
    
    bsdf = bs_df(df)
    print(bsdf)