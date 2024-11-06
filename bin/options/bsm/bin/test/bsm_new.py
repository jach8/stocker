############################################################################################################################################
# Black scholes Merton (1973) model for pricing European call options & Put Valuation

############################################################################################################################################

import numpy as np 
import pandas as pd 
from scipy.stats import norm 
from scipy.optimize import minimize 
from scipy.integrate import quad 


def dN(x):
    ''' PDF of Standard Normal: ~ N(0, 1)'''
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

def N(d):
    ''' CDF of Standard Normal: ~ N(0, 1)'''
    return quad(lambda x: dN(x), -20, d, limit= 50)[0]

def d1(S, K, T, r, sigma):
    ''' Black Scholes Merton (1973) d1 component'''
    d_0 = np.log(S/K)
    d_1 = T* (r + (0.5 * sigma ** 2)) * 1/(sigma * np.sqrt(T))
    return d_0 + d_1

def call_value(S, K, T, r, sigma):
    ''' Black Scholes European Call Option Valuation'''
    d_1 = d1(S, K, T, r, sigma)
    d_2 = d_1 - sigma * np.sqrt(T)
    return S * N(d_1) - np.exp(-r * T) * K * N(d_2)

def put_value(S, K, T, r, sigma):
    ''' Black Scholes European Put Option Valuation'''
    return call_value(S, K, T, r, sigma) - S + np.exp(-r * T) * K

def call_delta(S, K, T, r, sigma):
    ''' Black Scholes European Call Option Delta'''
    d_1 = d1(S, K, T, r, sigma)
    return N(d_1)

def gamma(S, K, T, r, sigma):
    ''' Black Scholes European Call Option Gamma'''
    d_1 = d1(S, K, T, r, sigma)
    return dN(d_1) / (S * sigma * np.sqrt(T))

def call_theta(S, K, T, r, sigma):
    ''' Black Scholes European Call Option Theta'''
    d_1 = d1(S, K, T, r, sigma)
    d_2 = d_1 - sigma * np.sqrt(T)
    t0 = -((S * dN(d_1) * sigma) / (2 * np.sqrt(T)))
    t1 = r * K * np.exp(-r * T) * N(d_2)
    return (t0 + t1) / 252

def call_rho(S, K, T, r, sigma):
    ''' Returns the change in option price for each 1% change in interest rate'''
    d_1 = d1(S, K, T, r, sigma)
    d_2 = d_1 - sigma * np.sqrt(T)
    return (K * T * np.exp(-r * T) * N(d_2)) * 0.01

def vega(S, K, T, r, sigma):
    ''' the change in the option price for each 1% change in volatility'''
    d_1 = d1(S, K, T, r, sigma)
    return (S * dN(d_1) * np.sqrt(T)) * 0.01

def call_option(S, K, T, r, sigma):
    ''' Black Scholes European Call Option '''
    return {
        'Value': call_value(S, K, T, r, sigma),
        'Vega': vega(S, K, T, r, sigma),
        'Delta': call_delta(S, K, T, r, sigma),
        'Gamma': gamma(S, K, T, r, sigma),
        'Theta': call_theta(S, K, T, r, sigma),
        'Rho': call_rho(S, K, T, r, sigma)
    }


# def plot_greeks(function, greek):
#     # Model Parameters
#     St = 100.0  # index level
#     r = 0.05  # risk-less short rate
#     sigma = 0.2  # volatility
#     t = 0.0  # valuation date

#     # Greek Calculations
#     tlist = np.linspace(0.01, 1, 25)
#     klist = np.linspace(80, 120, 25)
#     V = np.zeros((len(tlist), len(klist)), dtype=np.float)
#     for j in range(len(klist)):
#         for i in range(len(tlist)):
#             V[i, j] = function(St, klist[j], t, tlist[i], r, sigma)

#     # 3D Plotting
#     x, y = np.meshgrid(klist, tlist)
#     fig = plt.figure(figsize=(9, 5))
#     plot = p3.Axes3D(fig)
#     plot.plot_wireframe(x, y, V)
#     plot.set_xlabel('strike $K$')
#     plot.set_ylabel('maturity $T$')
#     plot.set_zlabel('%s(K, T)' % greek)



if __name__ == "__main__":
    print('---' * 20, '\n\n Black Scholes Merton (1973) Model for European Call Options \n\n')
    params = {'S': 446.81, 'K': 444.0, 'T': 0.03642094723691946, 'r': 0.04, 'sigma': 0.16309430664062502}
    S = params['S']
    K = params['K']
    T = params['T']
    r = params['r']
    sigma = params['sigma'] - 0.03

    print(call_option(S, K, T, r, sigma))