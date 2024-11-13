'''
Calibration of Jump Diffusion, and Mean reversion Simulation models 

'''

import numpy as np 
import pandas as pd 
import sqlite3 as sql 
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import mplfinance as mpf


def gbm(S0, r, days, mu, sigma, number_of_sims):
    ''' Geometric Brownian Motion without Drift 
        :   dS_t = d S_t dt + sigma S_t dW_t
    - Inputs: 
        : S0 = Initial Stock Price 
        : r = Risk Free Rate
        : days = Number of days to expiration
        : sigma = Implied Volatility
        : number_of_sims = Number of simulations to run
    - Outputs:
        : S = Stock Price Paths
    '''
    N = 100 # Time Steps
    T = days/252 # Number of years 
    dt = T/N # Each Time Step
    discount = np.exp(-r*T)
    # np.random.RandomState() # ensures that each sim is different.
    # # Precompute stock price paths. 
    # sigma = 1
    # S = np.zeros((days+1, number_of_sims))
    # S[0] = S0
    # for path in range(1, int(days+1)):
    #     Z = np.random.normal(size = number_of_sims)
    #     S[path] = S[path-1]*np.exp((r-0.5*sigma**2)*(days/252) + sigma*np.sqrt(days/252)*Z)
    # return S
    
    # simulation using numpy arrays
    St = np.exp(
    (mu - sigma ** 2 / 2) * dt
    + sigma * np.random.normal(0, np.sqrt(dt), size=(number_of_sims,N)).T
    )
    # include array of 1's
    St = np.vstack([np.ones(number_of_sims), St])
    # multiply through by S0 and return the cumulative product of elements along a given simulation path (axis=0). 
    St = S0 * St.cumprod(axis=0)
    return St

def poission_jump(S0, r, days, mu, sigma, number_of_sims, lam = 0.005, mj = 0.01, sj = 0.01):

    ''' Geometric Brownian Motion with Jump Diffusion 
        :   dS_t = d S_t dt + sigma S_t dW_t + S_t dJ_t
        
    - Inputs:
        : S0 = Initial Stock Price
        : r = Risk Free Rate
        : days = Number of days to expiration
        : sigma = Implied Volatility
        : lam = Jump Intensity (number of jumps per annum)
        : mj = Expected Jump size 
        : sj = Jump Size Volatility
        : number_of_sims = Number of simulations to run
    - Outputs:
        : S = Stock Price Paths
    '''
    N = number_of_sims
    np.random.RandomState() # ensures that each sim is different.
    T = days/252
    dt = T/252
    discount = np.exp(-r*T)
    S = np.zeros((int(days+1), N))
    S[0] = S0
    for path in range(1, days+1):
        Z = np.random.normal(size = N)
        S[path] = S[path-1]*np.exp((r-0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)
        jump = np.multiply(np.random.poisson(lam*dt, N), np.random.normal(mj, sj, N))
        S[path] = S[path] * (1 + 0.1 * jump)
    return S
    

def merton_jump(S0, r, days, mu, sigma, number_of_sims, lam = 0.005, m = 0.02, v = 0.003):
    """
    Merton Jump Diffusion Model
    :   dS_t = d S_t dt + sigma S_t dW_t + S_t dJ_t
    - Inputs:
        : S0 = Initial Stock Price
        : r = Risk Free Rate
        : days = Number of days to expiration
        : sigma = Implied Volatility
        : lam = Jump Intensity (number of jumps per annum)
        : m = Expected Jump size 
        : v = Jump Size Volatility
        : number_of_sims = Number of simulations to run
    - Outputs:
        : S = Stock Price Paths
    """
    np.random.RandomState() 

    N = number_of_sims
    T = days/252
    dt = T/252
    discount = np.exp(-r*T)
    size = (days+1, N)
    poi_rv = np.multiply(np.random.poisson(lam * dt, size = size), np.random.normal(m, v, size = size)).cumsum(axis = 0)
    geo = np.cumsum(((r - sigma**2/2 - lam * (m+ v**2 *0.5)) * dt + sigma * np.sqrt(dt) * np.random.normal(size = size )), axis = 0)
    S = np.exp(geo + poi_rv) * S0
    return S
    
    
def heston_path(S0, r, days, mu, sigma, number_of_sims, kappa, theta, v_0, rho, xi):
    ''' Price Paths using the Heston Stochastic Volatility Model 
        : S_t = S_t-1 exp((r - 0.5 v_t-1) dt + sqrt(v_t-1) dW_t)
        : v_t = v_t-1 + kappa(theta - v_t-1) dt + xi sqrt(v_t-1) dZ_t
    - Inputs:
        : kappa = Mean Reversion Factor
        : theta = Long Run Average Volatility
        : v_0 = Initial Volatility
        : rho = Correlation between the Brownian Motions
        : xi = Volatility of Volatility (Volatility Factor)
    - Outputs:
        : S = Stock Price Paths
    '''
    np.random.RandomState() 
    N = number_of_sims
    T = days/252
    dt = T/252
    discount = np.exp(-r*T)
    size = (N, days+1)
    prices = np.zeros(size)
    sigs = np.zeros(size)
    S_t = S0
    v_t = v_0
    cov_mat = np.array([[1, rho], [rho, 1]])
    for t in range(days+1):
        WT = np.random.multivariate_normal(np.array([0,0]), cov = cov_mat, size = N) * np.sqrt(dt)
        S_t = S_t * np.exp( (r - 0.5 * v_t) * dt + np.sqrt(v_t) * WT[:, 0] )
        v_t = np.abs(v_t + kappa * (theta - v_t ) * dt  + xi * np.sqrt(v_t) * WT[:, 1])
        prices[:, t] = S_t
        sigs[:, t] = v_t
    
    return prices.T

def est_vol(df, lookback=10):
    # Estimate Volatility of a stock. 
    """ 
    This is the Yang-Zheng (2005) Estimator for Valatlity; 
        Yang Zhang is a historical volatility estimator that handles 
            1. opening jumps
            2. the drift and has a minimum estimation error 
    """
    o = df.Open
    h = df.High
    l = df.Low
    c = df.Close
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


def heston_calibration(df, r, days, number_of_sims):
    ''' Calibrate the Heston Model to the Stock Prices '''
    def heston_error(params, S0, r, days, mu, sigma, number_of_sims, prices):
        kappa, theta, v_0, rho, xi = params
        S = heston_path(S0, r, days, mu, sigma, number_of_sims, kappa, theta, v_0, rho, xi)
        error = np.mean(np.abs(S - prices[-1]))
        return error
    
    np.random.RandomState() 
    S0 = df.Close.iloc[-1]
    sigma = est_vol(df).iloc[-1]
    mu = 0
    prices = df.Close.values
    params = np.array([0.001, 0.005, 0.002, 0.05, 0.01])
    result = minimize(heston_error, params, args = (S0, r, days, mu, sigma, number_of_sims, prices))
    return result.x


def simulate_stock(
    stock, 
    df,
    method = 'gbm',
    r = 0.0375,
    forecast_periods = 10,
    number_of_sims = 1000,
    **kwargs
    ):
    
    methods = {
        'gbm': gbm,
        'poission_jump': poission_jump,
        'merton_jump': merton_jump,
        'heston_path': heston_path
    }
    
    S0 = df.Close.iloc[-1]
    sigma = est_vol(df).mean()
    mu = df.Close.pct_change().mean()
    meth = methods[method]
    if method == 'heston_path':
        print(f"Stock: {stock} | Mu: {mu} | Sigma: {sigma}")
        kappa, theta, v_0, rho, xi = heston_calibration(df, r, forecast_periods, number_of_sims)
        S = meth(S0, r, forecast_periods, mu, sigma, number_of_sims, kappa, theta, v_0, rho, xi)
        iter = 0
        while S.shape[0] < forecast_periods:
            
            print(f"Stock: {stock} | Iteration: {iter} | Volatility: {sigma}")
            S = meth(S0, r, forecast_periods, mu, sigma, number_of_sims, kappa, theta, v_0, rho, xi)
            iter += 1
    else:
        S = meth(S0, r, forecast_periods, mu, sigma, number_of_sims, **kwargs)
    return S
    
    
def find_cases(s, sim_date, periods):
         # find the upper 25% and lower 25%
        dr = pd.date_range(start= sim_date, periods=periods, freq='D')
        u = pd.DataFrame(np.quantile(s, 0.75, axis=1), columns = ['Best'], index = dr)
        l = pd.DataFrame(np.quantile(s, 0.25, axis=1), columns = ['Worst'], index = dr)
        m = pd.DataFrame(np.quantile(s, 0.5, axis=1), columns = ['Expected'], index = dr)
        return pd.concat([u, m, l], axis = 1)
    
def plot_paths(stock,
    connection,
    r = 0.0375,
    forecast_periods = 100,
    number_of_sims = 1000,
    **kwargs):
    
    """
    Plot the 4 simulation methods, along with the three best case scenarios: 
        1st Scenario (best) is the mean of the upper 25% of the simulations
        2nd Scenario (best) is the mean of the middle 50% of the simulations
        3d Scenario (best) is the mean of the lower 25% of the simulations

    Args:
        stock (str): stock ticker
        connection (sql connection): connection to the database
        r (float, optional): _description_. Defaults to 0.0375.
        forecast_periods (int, optional): _description_. Defaults to 10.
        number_of_sims (int, optional): _description_. Defaults to 1000.
    """
    df = pd.read_sql(f"SELECT * FROM {stock}", connection, index_col = 'Date', parse_dates = ['Date'])
    meths = ['gbm', 'poission_jump', 'merton_jump', 'heston_path']
    d = {meth:simulate_stock(
        stock, 
        df, 
        method = meth,
        r = r,
        forecast_periods = forecast_periods,
        number_of_sims = number_of_sims,
        **kwargs) for meth in meths}

    date_range = pd.date_range(start = df.index[-1], periods = forecast_periods, freq = 'D')
    for meth in meths:
        d[meth] = pd.DataFrame(d[meth][1:, :], index = date_range)
    
    def plot_cases(fig, ax, s, title, ts):
        cases = find_cases(s, ts[-1], forecast_periods)
        ax.plot(cases.index, cases.Best, color='green', alpha=0.9, label = 'Best Case')
        ax.plot(cases.index, cases.Expected, color='black', alpha=0.9, label = 'Expected Case')
        ax.plot(cases.index, cases.Worst, color='red', alpha=0.9, label = 'Worst Case')
        return ax
    
    fig, ax = plt.subplots(2,2, figsize = (15, 10))
    ax = ax.flatten()
    for i, meth in enumerate(meths):
        # Candlestick plot
        mpf.plot(df.tail(20), ax = ax[i], type = 'candle', volume = False, show_nontrading = True)
        # ax[i].plot(d[meth], alpha = 0.1, color = 'grey')
        ax[i].set_title(meth)
        ax[i] = plot_cases(fig, ax[i], d[meth], meth, df.index)
        fig.autofmt_xdate()

    plt.show()
        
    
    
    
    
if __name__ == "__main__":
    import sys 
    sys.path.append('/Users/jerald/Documents/Dir/Python/Stocks')
    from main import Pipeline
    
    p = Pipeline()
    price_db = p.Pricedb.daily_db
    stocks = p.Earningsdb.upcoming_earnings(n = 6)
    for stock in stocks:
        plot_paths(stock, price_db)

    
    