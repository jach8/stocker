import numpy as np 
from scipy.stats import norm 
from scipy.optimize import brentq

class bsm(object):
    ''' General Class to obtain Greeks for European call/put options in BSM Model. 

    Attributes
    ==========
    S0 : float
        initial stock/index level
    K : float
        strike price
    T : float
        maturity (in year fractions)
    r : float
        constant risk-free short rate
    sigma : float
        volatility factor in diffusion term
    option_type : str
        'call' or 'put'
    price : float
        market price of the option
    '''

    def __init__(self, S0, K, T, r, sigma, option_type, price, q = 0):
        self.S0 = float(S0)
        self.K = K
        self.T = T
        self.r = r
        self.q = q
        self.sigma = sigma
        self.type = option_type
        self.price = price


    def N(self, x):
        return norm.cdf(x)
    

    def params(self):
        ''' Convinience function to collect parameters. '''
        return {'S': self.S0, 'K': self.K, 'T': self.T, 'r': self.r, 'sigma': self.sigma}
    
    def d1(self):
        ''' Helper function. '''
        d1 = ((np.log(self.S0 / self.K)+ (self.r - self.q + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T)))
        return d1
    
    def d2(self):
        return self.d1() - self.sigma * np.sqrt(self.T)
    
    def _call_value(self):
        return self.S0 * np.exp(-self.q * self.T) * self.N(self.d1()) - self.K * np.exp(-self.r * self.T) * self.N(self.d2())
    
    def _put_value(self):
        return self.K * np.exp(-self.r * self.T) * self.N(-self.d2()) - self.S0 * np.exp(-self.q * self.T) * self.N(-self.d1())
    
    def _call_greeks(self):
        call_vega = self.S0 * np.exp(-self.r * self.T) * norm.pdf(self.d1()) * np.sqrt(self.T)
        call_delta = self.N(self.d1()) * np.exp(-self.r * self.T)
        call_gamma = norm.pdf(self.d1()) / (self.S0 * self.sigma * np.sqrt(self.T))
        call_theta = self.S0 * np.exp(-self.r * self.T) * self.q * self.N(self.d1()) - self.K * np.exp(-self.r * self.T) * self.r * self.N(self.d2()) - self.S0 * np.exp(-self.r * self.T) * (self.sigma / (2 * np.sqrt(self.T))) * norm.pdf(self.d1())
        call_rho = self.K * self.T * np.exp(-self.r * self.T) * self.N(self.d2())
        return call_vega, call_delta, call_gamma, call_theta, call_rho
    
    def _put_greeks(self):
        put_vega = self.S0 * np.exp(-self.r * self.T) * norm.pdf(self.d1()) * np.sqrt(self.T)
        put_delta = self.N(-self.d1()) * - np.exp(-self.r * self.T) 
        put_gamma = (norm.pdf(self.d1()) * np.exp(-self.r * self.T)) / (self.S0 * self.sigma * np.sqrt(self.T))
        put_theta = -self.S0 * np.exp(-self.r * self.T) * self.q * self.N(-self.d1()) + self.K * np.exp(-self.r * self.T) * self.r * self.N(-self.d2()) + self.S0 * np.exp(-self.r * self.T) * (self.sigma / (2 * np.sqrt(self.T))) * norm.pdf(-self.d1())
        put_rho = -self.K * self.T * np.exp(-self.r * self.T) * self.N(-self.d2())
        return put_vega, put_delta, put_gamma, put_theta, put_rho


    def fairvalue(self):
        if self.type.lower() == 'call':
            vega, delta, gamma, theta, rho = self._call_greeks()
            return {'fairvalue': self._call_value(), 'vega': vega, 'delta': delta, 'gamma': gamma, 'theta': theta, 'rho': rho}
        elif self.type.lower() == 'put':
            vega, delta, gamma, theta, rho = self._put_greeks()
            return {'fairvalue': self._put_value(), 'vega': vega, 'delta': delta, 'gamma': gamma, 'theta': theta, 'rho': rho}

    


if __name__ == '__main__':
    K = 100
    r = 0.1
    T = 0.5
    sigma = 0.3
    S = 100
    lastprice = 10
    b = bsm(S, K, T, r, sigma, 'put', lastprice, )
    print(b.fairvalue())