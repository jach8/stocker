
import numpy as np
from scipy.stats import norm, lognorm, t
from numpy.polynomial import Polynomial
import pandas as pd 
import math

class OptionSim:
    def __init__(self, S0, K, r, days, timevalue, sigma, option_type, number_of_sims, Observed = None):
        """ 
        
            Initialize the OptionSim class which implements the Longstaff-Schwartz Method for Pricing American Options
            The ALgorithm is as follows:
            1. Generate Stock Price Paths
                Follows a Geometric Brownian Motion without Drift
                    S_t = S_{t-1} * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
            2. Calculate the payoff of the option at expiration
            3. Iterate backwards in time, and approximate the continuation value using regression
                Discount the continuation value to the present value
            4. If the exercise value is greater than the continuation value, exercise the option
            5. Calculate the expected value of the option at time t
            6. Repeat the process for all paths
        
        Inputs: 
            S0: Initial Stock Price
            K: Strike Price
            r: Risk Free Rate
            days: Number of days to expiration
            sigma: Implied Volatility
            option_type: 'Call' or 'Put'
            number_of_sims: Number of simulations to run
            Observed: Observed price of option, if available
            
        Methods:
            stock_paths: Generate Stock Price Paths
            black_scholes_analytical: Calculate the Black-Scholes Analytical Price of the Option
            mc_sim: Calculate the Monte Carlo Price of the Option
            payoff: Calculate the payoff of the option at time t
            itm_select: Select in-the-money options
            discount_function: Calculate the discount factor
            fit_quad: Fit a polynomial to the cashflows
            l_poly: Fit a Laguerre polynomial to the cashflows
            control_variate: Approximate the conditional payoff using a control variate
            gaussian_basis: Gaussian Basis Function
            design_matrix: Create a design matrix with basis functions
            gaussian_basis_fit: Fit Gaussian Basis Functions to the cashflows
            longstaff_schwartz_iter: Longstaff-Schwartz Iteration
            ls: Longstaff-Schwartz Method
            ls_normal: Longstaff-Schwartz Method with Gaussian Basis Functions
            ls_cv: Longstaff-Schwartz Method with Control Variate
            run: Run the Longstaff-Schwartz Method
            run_ir: Run the Longstaff-Schwartz Method with Independent Replications
            run_ir2: Run the Longstaff-Schwartz Method with Independent Replications
        
        
        
        """
        self.S0 = S0
        self.K = K
        self.r = r
        self.days = np.maximum(int(days), 1)
        self.T = timevalue
        self.sigma = sigma
        self.N = number_of_sims
        self.dt = self.T/self.days
        self.option_type = option_type.lower()
        self.discount = np.exp(-self.r*self.T)
        self.Observed = Observed

    def stock_paths(self):
        """
        Geometry Brownian Motion without Drift or Jump Diffusion Process
        """
        # initialize random seed
        # r_int = np.random.randint(8128)
        # np.random.seed(r_int) 
        # np.random.seed(0)
        np.random.RandomState() # ensures that each sim is different.
        # Precompute stock price paths. 
        S = np.zeros((self.days+1, self.N))
        S[0] = self.S0
        for path in range(1, int(self.days+1)):
            Z = np.random.normal(size = self.N)
            S[path] = S[path-1]*np.exp((self.r-0.5*self.sigma**2)*self.dt + self.sigma*np.sqrt(self.dt)*Z)
        return S

    def stock_path_jump(self):
        """ 
        Jump Diffusion Process for simulating stock price paths 
            The Jump Diffusion Process is given by a Poisson Process with intensity λ
                S_t = S_{t-1} * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z) * (1 + λ * N(0,1))
                Where λ denotes the intensity of the poisson process, and N(0,1) is a standard normal random variable
                

        Returns:
            numpy array: Stock Price Paths
        """
        
        np.random.RandomState()
        lambda_ = 0.5
        S = np.zeros((int(self.days+1), self.N))
        S[0] = self.S0
        for path in range(1, self.days+1):
            Z = np.random.normal(size = self.N)
            S[path] = S[path-1]*np.exp((self.r-0.5*self.sigma**2)*self.dt + self.sigma*np.sqrt(self.dt)*Z)
            jump = np.random.poisson(lambda_*self.dt, self.N)
            S[path] = S[path] * (1 + 0.1 * jump)
        return S

    def black_scholes_analytical(self):
        """
        Calculate the Black-Scholes Analytical Price of the Option
        The Black-Scholes Formula is given by:
            Call: S0*N(d1) - K*exp(-rT)*N(d2)
            Put: K*exp(-rT)*N(-d2) - S0*N(-d1)
        Where:
            d1 = (ln(S0/K) + (r + 0.5*sigma^2)*T)/(sigma*sqrt(T))
            d2 = d1 - sigma*sqrt(T)
        Returns:
            float: Black-Scholes Analytical Price of the Option
        """
        d1 = (np.log(self.S0/self.K) + (self.r + 0.5*self.sigma**2)*self.T)/(self.sigma*np.sqrt(self.T))
        d2 = d1 - self.sigma*np.sqrt(self.T)
        if self.option_type == 'call':
            return self.S0*norm.cdf(d1) - self.K*self.discount*norm.cdf(d2)
        elif self.option_type == 'put':
            return self.K*self.discount*norm.cdf(-d2) - self.S0*norm.cdf(-d1)

    def mc_sim(self,jump = False):
        """
        Returns the Monte Carlo Pricing method of the option 
        The Monte Carlo Method is given by:
            1. Generate Stock Price Paths
            2. Calculate the payoff of the option at expiration
            3. Discount the payoff to the present value
            4. Repeat the process for all paths
            5. Return the mean of the discounted payoffs

        Args:
            jump (bool, optional): Use Jump Diffusion Process or not. Defaults to False.

        Returns:
            _type_: float: Monte Carlo Price of the Option
        """
        if jump == False:
            S = self.stock_paths()
        else:
            S = self.stock_path_jump()
        if self.option_type == 'call':
            payoff = np.maximum(S[-1]-self.K, 0)
        elif self.option_type == 'put':
            payoff = np.maximum(self.K-S[-1], 0)
        return self.discount*np.mean(payoff)
    
    def lsm_primal_valuation(self, jump = False):
        """
        Returns the Longstaff-Schwartz Method for Pricing American Options
        The Algorithm is as follows:
        1. Generate Stock Price Paths
            Follows a Geometric Brownian Motion without Drift
                S_t = S_{t-1} * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
        2. Calculate the payoff of the option at expiration
        3. Iterate backwards in time, and approximate the continuation value using regression
            Discount the continuation value to the present value
        4. If the exercise value is greater than the continuation value, exercise the option
        5. Calculate the expected value of the option at time t
        6. Repeat the process for all paths

        Returns:
            float: Longstaff-Schwartz Price of the Option
        """
        # Stock Price Paths 
        if jump == False:
            S = self.stock_paths()
        else:
            S = self.stock_path_jump()
        
        # Inner Values 
        h = np.maximum(self.K - S, 0)
        
        # Initialize Present Value Vector
        V = h[-1]
        
        # American Option Valuation by Backwards Induction
        for t in range(self.days - 1, 0, -1):
            rg = np.polyfit(S[t], V * self.discount, 5)
            C = np.polyval(rg, S[t])
            V = np.where(h[t] > C, h[t], V * self.discount)
        V0 = self.discount * np.sum(V) / self.N
        return V0

    def payoff(self, x):
        """
        Calculate the payoff of the option at time t
        Args:
            x (numpy array): Stock Price Paths at time t
        Returns:
            numpy array: Payoff of the option at time t
        """
        
        if self.option_type == 'call':
            return np.maximum(x - self.K, 0.0)
        elif self.option_type == 'put':
            return np.maximum(self.K - x, 0.0)

    def itm_select(self, x):
        """
        Select in-the-money options
        Args:
            x (numpy array): Stock Price Paths at time t
        Returns:
            numpy array: Boolean index of in-the-money options
        """
        return x > 0

    def discount_function(self, t0, t1):
        """
        Calculate the discount factor between two time periods
        Args:
            t0 (float): Time t0
            t1 (float): Time t1
        Returns:
            float: Discount Factor
        """
        
        return np.exp(-self.r * (t1 - t0))
    
    def fit_quad(self, x, y, deg = 3):
        """
        Fit a polynomial to the cashflows
        Args:
            x (numpy array): Stock Price Paths at time t
            y (numpy array): Cashflows
            deg (int, optional): Degree of the polynomial. Defaults to 3.
        Returns:
            Polynomial: Polynomial fit
        """

        vars = np.array([x**i for i in range(deg+1)])
        coef = np.linalg.lstsq(vars.T, y, rcond=None)[0]
        return Polynomial(coef)

    def l_poly(self, x, y, n = 2):
        """
        Fit a Laguerre polynomial to the cashflows this is used as a basis function for the LSM regression method

        Args:
            x (numpy array): Stock Price Paths at time t
            y (numpy array): Cashflows
            n (int, optional): Degree of the polynomial. Defaults to 2.
        Returns:
            _type_: Polynomial: Polynomial fit
        """
        # Laguerre polynomial basis functions 
        # L0 = exp(-x/2)
        # L1 = exp(-x/2) * (1 - x)
        # L2 = exp(-x/2) * (1 - 2*x + x**2/2)
        # Ln = exp(-x/2) * exp(X)/ n!  * nth derivative of (X**n * exp(-X))

        l0 = np.exp(-x/2)
        l1 = np.exp(-x/2) * (1 - x)
        l2 = np.exp(-x/2) * (1 - 2*x + x**2/2)
        l3 = np.exp(-x/2) * (1 - 3*x + 3*x**2/2 - x**3/6)
        vars = np.array([l0, l1, l2, l3])
        coef = np.linalg.lstsq(vars.T, y, rcond=None)[0]
        return Polynomial(coef)


    def control_variate(self, x, y):
        """
        Approximate the conditional payoff using a control variate (Variance Reduction Technique)

        Args:
            x (numpy array): array of stock prices
            y (numpy array): array of cashflows

        Returns:
            _type_: lambda: function that takes x and returns the fitted value
        """
        # approximate the conditional payoff using control variate
        # xbar - beta (y - ybar)
        xbar = np.mean(x)
        ybar = np.mean(y)
        beta = np.sum((x - xbar)*(y - ybar))/np.sum((x - xbar)**2)
        return lambda x: x - beta*(y - ybar)

    def gaussian_basis(self, x, mu, sigma):
        """
        Gaussian Basis Function
        Args:
            x (numpy array): Stock Price Paths at time t
            mu (float): Mean of the Gaussian Basis Function
            sigma (float): Standard Deviation of the Gaussian Basis Function
            
        Returns:
            numpy array: Gaussian Basis Function
        """
        #return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi)) # Gaussian Basis Function
        return lognorm.pdf(x, s=sigma, scale=np.exp(mu)) # Lognormal Basis Function

    def design_matrix(self, x, deg = 3):
        """
        Create a design matrix with basis functions
        Args:
            x (numpy array): Stock Price Paths at time t
            deg (int, optional): Degree of the polynomial. Defaults to 3.
        Returns:
            numpy array: Design Matrix with Basis Functions
        """
        # create design matrix with basis functions
        X = np.zeros((len(x), deg))
        if len(X)>0:
            for i in range(deg):
                mu = np.linspace(min(x), max(x), deg)[i]
                sigma = (max(x) - min(x)) / (deg * 2)
                X[:, i] = self.gaussian_basis(x, mu, sigma)
        return X

    def gaussian_basis_fit(self, x, y):
        """
        Fit Gaussian Basis Functions to the cashflows
        Args:
            x (numpy array): Stock Price Paths at time t
            y (numpy array): Cashflows
        Returns:
            _type_: lambda: function that takes x and returns the fitted value
        """
        
        # fit gaussian basis functions to cashflows
        xvars = self.design_matrix(x)
        coef = np.linalg.lstsq(xvars, y, rcond=None)[0]
        # return a function that takes x and returns the fitted value
        return lambda x: np.dot(self.design_matrix(x), coef)

    
    def longstaff_schwartz_iter(self, X, t, fit):
        """
        Longstaff-Schwartz Iteration for the LSM method. 
        This method Yields the cashflows, stock price paths, fitted polynomial, continuation value, exercise value, and boolean index of early exercise
        Args:
            X (numpy array): Stock Price Paths
            t (numpy array): Time Periods
            fit (function): Function to fit the cashflows
        Yields:
            numpy array: Cashflows
            numpy array: Stock Price Paths
            Polynomial: Fitted Polynomial
            numpy array: Continuation Value
            numpy array: Exercise Value
            numpy array: Boolean Index of Early Exercise
        """
        # given no prior exercise we just receive the final payoff
        cashflow = self.payoff(X[-1, :])
        # iterating backwards in time
        for i in reversed(range(1, X.shape[0])):
            # discount cashflows from next period
            cashflow = cashflow * self.discount_function(t[i], t[i + 1])
            x = X[i, :]
            # exercise value for time t[i]
            exercise = self.payoff(x)
            # boolean index of all in-the-money paths, choose path with payoff > 0 
            itm = self.itm_select(x)
            # fit curve
            fitted = fit(x[itm], cashflow[itm])
            # approximate continuation value
            continuation = fitted(x)
            # boolean index where exercise is beneficial
            ex_idx = itm & (exercise > continuation)
            # update cashflows wiconth early exercises
            cashflow[ex_idx] = exercise[ex_idx]

            yield cashflow, x, fitted, continuation, exercise, ex_idx

    def ls(self, deg = 3, jump = False, ts = None):
        """
        Longstaff-Schwartz Method
        Args:
            deg (int, optional): Degree of the polynomial. Defaults to 3.
            jump (bool, optional): Use Jump Diffusion Process or not. Defaults to False.
            ts (int, optional): Number of time steps. Defaults to None. 
        Returns:
            float: Longstaff-Schwartz Price of the Option
        """
        if ts == None:
            ts = self.N
        if jump == False:
            X = self.stock_paths()
        else:
            X = self.stock_path_jump()
        t = np.linspace(0, self.days, self.N *ts)
        for cashflow, *_ in self.longstaff_schwartz_iter(X, t, self.fit_quad):
            pass
        return cashflow.mean(axis = 0) * self.discount_function(t[0], t[1])

    def ls_normal(self, deg = 3, jump = False, ts = None):
        """"
        Longstaff-Schwartz Method with Gaussian Basis Functions
        """
        if ts == None:
            ts = self.N
        if jump == False:
            X = self.stock_paths()
        else:
            X = self.stock_path_jump()
        t = np.linspace(0, self.days, self.N *ts)
        for cashflow, *_ in self.longstaff_schwartz_iter(X, t, self.gaussian_basis_fit):
            pass
        return cashflow.mean(axis=0) * self.discount_function(t[0], t[1])

    def ls_cv(self, deg = 3, jump = False, ts = None):
        """
        Longstaff-Schwartz Method with a Control Variate
        """
        
        if ts == None:
            ts = self.N
        if jump == False:
            X = self.stock_paths()
        else:
            X = self.stock_path_jump()
        t = np.linspace(0, self.days, self.N *ts)
        for cashflow, *_ in self.longstaff_schwartz_iter(X, t, self.control_variate):
            pass
        return cashflow.mean(axis=0) * self.discount_function(t[0], t[1])
    
    def run(self, jump = False):
        """
        Run the Longstaff-Schwartz Method
        Args:
            jump (bool, optional): Use Jump Diffusion Process or not. Defaults to False.
        Returns:
            pandas DataFrame: Longstaff-Schwartz Price of the Option
        """
        lsmc_polyfit = self.ls(jump = jump)
        lcv = self.ls_cv(jump = jump)
        lsmc_normal = self.ls_normal(jump = jump)
        mc_fair_price = self.mc_sim(jump = jump)
        bs_call = self.black_scholes_analytical()
        lsm_primal = self.lsm_primal_valuation(jump = jump )
        if self.Observed == None:
            return pd.DataFrame({
                'LSMC Normal': lsmc_normal,
                'LSMC Poly': lsmc_polyfit,
                'LSMC CV': lcv,
                "LSM Primal": lsm_primal,
                'MC': mc_fair_price,
                'BS': bs_call
            }, index = [0])
        else:
            return pd.DataFrame({
                'LSMC Normal': lsmc_normal,
                'LSMC Poly': lsmc_polyfit,
                'LSMC CV': lcv,
                "LSM Primal": lsm_primal,
                'MC': mc_fair_price,
                'BS': bs_call,
                'Observed': self.Observed
            }, index = [0]) 

    def run_ir(self, jump = False, replications = 10, alpha = 0.05):
        """
        Run the Longstaff-Schwartz Method with Independent Replications
        Args:
            jump (bool, optional): Use Jump Diffusion Process or not. Defaults to False.
            replications (int, optional): Number of replications. Defaults to 10.
            alpha (float, optional): Significance Level. Defaults to 0.05.
        Returns:
            pandas DataFrame: Longstaff-Schwartz Price of the Option
        """
        
        # Method of Indpendent Replications
        lsmc_poly_rep = [self.ls(jump) for _ in range(replications)]
        lsmc_cv_rep = [self.ls_cv(jump) for _ in range(replications)]
        lsmc_normal_rep = [self.ls_normal(jump) for _ in range(replications)]
        mc_rep = [self.mc_sim(jump) for _ in range(replications)]
        lsm_primal_rep = [self.lsm_primal_valuation(jump) for _ in range(replications)]

        # Grand Sample Mean Z_bar
        lsmc_poly = np.mean(lsmc_poly_rep)
        lsmc_cv = np.mean(lsmc_cv_rep)
        lsmc_normal = np.mean(lsmc_normal_rep)
        mc = np.mean(mc_rep)
        lsm_primal = np.mean(lsm_primal_rep)
        

        # Sample Variance 
        lsmc_poly_var = (1/(replications -1)) * np.var(lsmc_poly_rep)
        lsmc_cv_var = (1/(replications -1)) * np.var(lsmc_cv_rep)
        lsmc_normal_var = (1/(replications -1)) * np.var(lsmc_normal_rep)
        mc_var = (1/(replications -1)) * np.var(mc_rep)
        lsm_primal_var = (1/(replications -1)) * np.var(lsm_primal_rep)

        # Standard Error
        lsmc_poly_se = np.sqrt(lsmc_poly_var/replications)
        lsmc_cv_se = np.sqrt(lsmc_cv_var/replications)
        lsmc_normal_se = np.sqrt(lsmc_normal_var/replications)
        mc_se = np.sqrt(mc_var/replications)
        lsm_primal_se = np.sqrt(lsm_primal_var/replications)

        # t-statistic, and df 
        df = replications - 1
        t_stat = t.ppf(1-alpha , df)

        # Confidence Interval
        lsmc_poly_ci = [lsmc_poly - t_stat*np.sqrt(lsmc_poly_var / replications), lsmc_poly + t_stat*np.sqrt(lsmc_poly_var / replications)]
        lsmc_cv_ci = [lsmc_cv - t_stat*np.sqrt(lsmc_cv_var / replications), lsmc_cv + t_stat*np.sqrt(lsmc_cv_var / replications)]
        lsmc_normal_ci = [lsmc_normal - t_stat*np.sqrt(lsmc_normal_var / replications), lsmc_normal + t_stat*np.sqrt(lsmc_normal_var / replications)]
        mc_ci = [mc - t_stat*np.sqrt(mc_var / replications), mc + t_stat*np.sqrt(mc_var / replications)]
        lsm_primal_ci = [lsm_primal - t_stat*np.sqrt(lsm_primal_var / replications), lsm_primal + t_stat*np.sqrt(lsm_primal_var / replications)]

        bs = self.black_scholes_analytical()
        # return a row of a dataframe, with the CI as a list
        out = pd.DataFrame({
            'LSMC Normal μ': lsmc_normal,
            'LSMC Poly μ': lsmc_poly,
            'LSMC CV μ': lsmc_cv,
            'LSM Primal μ': lsm_primal,
            'MC μ': mc,
            'BS': bs,
            # 'LSMC Poly SE': lsmc_poly_se,
            # 'LSMC Normal SE': lsmc_normal_se,
            # 'LSMC CV SE': lsmc_cv_se,
            # 'MC SE': mc_se,
            # 'LSMC Normal CI': [(lsmc_normal_ci)],
            # 'LSMC Poly CI': [(lsmc_poly_ci)],
            # 'LSMC CV CI': [(lsmc_cv_ci)],
            # 'MC CI': [(mc_ci)],
        }, index = [0])

        if self.Observed == None:
            return out
        else:
            out['Observed'] = self.Observed
            return out    
    

if __name__ == "__main__":
    """ 
        Inputs: 
        S0: Initial Stock Price
        K: Strike Price
        r: Risk Free Rate
        days: Number of days to expiration
        sigma: Implied Volatility
        option_type: 'Call' or 'Put'
        number_of_sims: Number of simulations to run
        Observed: Observed price of option, if available
        S0, K, r, days, timevalue, sigma, option_type, number_of_sims, Observed = None
    """
    print("\n\n\nRunning...\n")
    S0 = [187.87, 187.87, 187.87]
    K = [190.0, 185.0, 187.5]
    days = [14, 14, 14]
    time_value = [14/365, 14/365, 14/365]
    sigma = [0.19373364868164064, 0.17591156127929689, 0.1506432592773438]
    otype = ['Call', 'Put', 'Put']
    Observed = [2.08, 1.52, 2.43]
    
    number_of_sims = [100, 100, 100]
    r = [0.05, 0.05, 0.05]

    for i in zip(S0, K, r, days, time_value, sigma, otype, number_of_sims, Observed):
        print(i)
        sim = OptionSim(i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7])
        print(sim.run())
        print('\n')
    
    print("\n")
