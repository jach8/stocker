''' 
Stock Price Simulation using Geometric Brownian Motion. 
    :Base Class for simulating the stock price using Geometric Brownian Motion 
    : Add option to incorporate the jump diffusion model following a Poisson Process 
    : Add option to incorporate the stochastic volatility model 
    : Add the option to incorporate the mean reversion model.

'''

import numpy as np 
import math 
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import minimize

mpl.rcParams['font.family'] = 'serif'

class OptionSim:
    def __init__(self, S0, r, days, sigma, number_of_sims):
        """ 
        Inputs: 
        S0: Initial Stock Price
        r: Risk Free Rate
        days: Number of days to expiration
        sigma: Implied Volatility
        number_of_sims: Number of simulations to run
        """
        self.S0 = S0
        self.r = r
        self.days = days
        self.sigma = sigma
        self.N = number_of_sims
        self.T = days/252
        self.dt = self.T/self.days
        self.discount = np.exp(-self.r*self.T)

    def stock_paths(self):
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
        np.random.RandomState() # ensures that each sim is different.
        # Precompute stock price paths. 
        S = np.zeros((self.days+1, self.N))
        S[0] = self.S0
        for path in range(1, int(self.days+1)):
            Z = np.random.normal(size = self.N)
            S[path] = S[path-1]*np.exp((self.r-0.5*self.sigma**2)*self.dt + self.sigma*np.sqrt(self.dt)*Z)
        return S

    def stock_path_jump(self, lam = 0.5, mj = 0.1, sj = 0.1):
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
        S = np.zeros((int(self.days+1), self.N))
        S[0] = self.S0
        for path in range(1, self.days+1):
            Z = np.random.normal(size = self.N)
            S[path] = S[path-1]*np.exp((self.r-0.5*self.sigma**2)*self.dt + self.sigma*np.sqrt(self.dt)*Z)
            jump = np.multiply(np.random.poisson(lam*self.dt, self.N), np.random.normal(mj, sj, self.N))
            S[path] = S[path] * (1 + 0.1 * jump)
        return S
    
    def merton_jump(self, lam = 0.5, m = 0, v = 0.3):
        size = (self.days+1, self.N)
        poi_rv = np.multiply(np.random.poisson(lam * self.dt, size = size), np.random.normal(m, v, size = size)).cumsum(axis = 0)
        geo = np.cumsum(((self.r - self.sigma**2/2 - lam * (m+ v**2 *0.5)) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size = size )), axis = 0)
        S = np.exp(geo + poi_rv) * self.S0
        return S
    
    def heston_path(self, kappa, theta, v_0, rho, xi):
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
        size = (self.N, self.days+1)
        prices = np.zeros(size)
        sigs = np.zeros(size)
        S_t = self.S0
        v_t = v_0
        cov_mat = np.array([[1, rho], [rho, 1]])
        for t in range(self.days+1):
            WT = np.random.multivariate_normal(np.array([0,0]), cov = cov_mat, size = self.N) * np.sqrt(self.dt)
            S_t = S_t * np.exp( (self.r - 0.5 * v_t) * self.dt + np.sqrt(v_t) * WT[:, 0] )
            v_t = np.abs(v_t + kappa * (theta - v_t ) * self.dt  + xi * np.sqrt(v_t) * WT[:, 1])
            prices[:, t] = S_t
            sigs[:, t] = v_t
        
        return prices.T
    

if __name__ == '__main__':
    print('To succeed in life, you need two things: ignorance and confidence. - Mark Twain')
    os = OptionSim(S0 = 100, r = 0.02, days = 255, sigma = 0.2, number_of_sims=1000)
    gbm = os.stock_paths()
    jump = os.stock_path_jump(lam = 1, mj = 0, sj = 0.3)
    merton = os.merton_jump(lam = 1, m = 0, v = 0.3)
    hest = os.heston_path(kappa = 4, theta = 0.02, v_0=0.02, xi = 0.9, rho = -0.8)

    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    ax[0].plot(gbm[:, :10], lw=1.5)
    ax[0].set_title('Geometric Brownian Motion')
    ax[1].plot(merton[:, :10], lw=1.5)
    ax[1].set_title('Merton Jump Diffusion')
    ax[2].plot(hest[:, :10], lw=1.5)    
    ax[2].set_title('Heston Stochastic Volatility')
    plt.show()

    