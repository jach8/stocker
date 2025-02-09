"""
Option Chain Density Analysis Module

This module provides tools for analyzing option chain density patterns through various
statistical distributions and kernel density estimation methods. It focuses on
processing open interest and volume data to calculate density metrics per expiration date.
"""

from typing import Dict, List, Tuple, Union, Optional
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import logging
from dataclasses import dataclass
import datetime as dt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DensityEstimationResult:
    """Container for density estimation results."""
    distribution_name: str
    parameters: Tuple
    pdf_values: np.ndarray
    goodness_of_fit: float
    strike_prices: np.ndarray

class OptionChainDensity:
    """
    A class for analyzing option chain density patterns and probability distributions.
    
    This class provides methods for:
    - Processing and validating option chain data
    - Fitting various probability distributions to the data
    - Calculating kernel density estimates
    - Visualizing density patterns with professional-grade plots
    """

    def __init__(self, option_chain: pd.DataFrame):
        """
        Initialize the OptionChainDensity analyzer.

        Args:
            option_chain (pd.DataFrame): DataFrame containing option chain data with columns:
                - strike: Strike prices
                - type: Option type (Call/Put)
                - openinterest: Open interest values
                - volume: Trading volume
                - expiry: Expiration dates

        Raises:
            ValueError: If required columns are missing or data is invalid
        """
        self._validate_input_data(option_chain)
        self.option_chain = option_chain
        self.logger = logging.getLogger(__name__)
        self._process_option_chain()

    def _validate_input_data(self, df: pd.DataFrame) -> None:
        """
        Validate input data for required columns and data types.

        Args:
            df (pd.DataFrame): Input option chain data

        Raises:
            ValueError: If validation fails
        """
        required_columns = ['strike', 'type', 'openinterest', 'volume', 'expiry']
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        if not all(df['type'].isin(['Call', 'Put'])):
            raise ValueError("Option type must be either 'Call' or 'Put'")

        try:
            pd.to_datetime(df['expiry'])
        except Exception as e:
            raise ValueError(f"Invalid expiry date format: {e}")

    def _process_option_chain(self) -> None:
        """Process and prepare option chain data for analysis."""
        logger.info("Processing option chain data...")
        
        # Add days to expiration
        self.option_chain['dte'] = (
            pd.to_datetime(self.option_chain['expiry']) - 
            pd.to_datetime(self.option_chain['expiry'].min())
        ).dt.days

        # Create aggregated views
        self.strikes = self.option_chain['strike'].unique()
        self.expiry_dates = pd.to_datetime(self.option_chain['expiry'].unique())
        
        # Calculate total density metrics
        self.total_oi = self._calculate_density_metric('openinterest')
        self.total_volume = self._calculate_density_metric('volume')

    def _calculate_density_metric(self, metric: str) -> pd.Series:
        """
        Calculate density metric (OI or volume) aggregated by strike.

        Args:
            metric (str): Column name to aggregate ('openinterest' or 'volume')

        Returns:
            pd.Series: Aggregated metric by strike
        """
        return self.option_chain.groupby('strike')[metric].sum()

    def fit_distributions(self) -> Dict[str, DensityEstimationResult]:
        """
        Fit multiple probability distributions to the option chain data.

        Returns:
            Dict[str, DensityEstimationResult]: Results for each distribution type
        """
        logger.info("Fitting probability distributions...")
        
        distributions = {
            'normal': stats.norm,
            'gamma': stats.gamma,
            'lognormal': stats.lognorm
        }
        
        results = {}
        X = self.total_oi.values.reshape(-1, 1)
        
        for name, dist in distributions.items():
            try:
                params = dist.fit(X)
                pdf_values = dist.pdf(self.strikes, *params)
                
                # Calculate goodness of fit using KS test
                _, p_value = stats.kstest(X.ravel(), dist.cdf, args=params)
                
                results[name] = DensityEstimationResult(
                    distribution_name=name,
                    parameters=params,
                    pdf_values=pdf_values,
                    goodness_of_fit=p_value,
                    strike_prices=self.strikes
                )
                
                logger.info(f"Successfully fitted {name} distribution (p-value: {p_value:.4f})")
                
            except Exception as e:
                logger.error(f"Failed to fit {name} distribution: {e}")
                continue
        
        return results

    def calculate_kde(self, bandwidth: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate Kernel Density Estimation for the option chain.

        Args:
            bandwidth (float): Bandwidth parameter for KDE

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (xx, yy, density)
        """
        logger.info(f"Calculating KDE with bandwidth {bandwidth}...")
        
        try:
            x = self.strikes
            y = self.total_oi.values
            
            xx, yy = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
            xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
            xy_train = np.vstack([y, x]).T
            
            kde = KernelDensity(bandwidth=bandwidth)
            kde.fit(xy_train)
            
            z = np.exp(kde.score_samples(xy_sample))
            z = np.reshape(z, xx.shape)
            
            return xx, yy, z
            
        except Exception as e:
            logger.error(f"KDE calculation failed: {e}")
            raise

    def plot_density_analysis(self, save_path: Optional[str] = None) -> None:
        """
        Create comprehensive density analysis visualization.

        Args:
            save_path (Optional[str]): Path to save the plot

        Returns:
            None
        """
        logger.info("Generating density analysis plot...")
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Option Chain Density Analysis', fontsize=16)
            
            # Plot 1: Distribution fits
            distributions = self.fit_distributions()
            for name, result in distributions.items():
                axes[0, 0].plot(
                    result.strike_prices,
                    result.pdf_values,
                    label=f'{name.capitalize()} (p={result.goodness_of_fit:.4f})'
                )
            axes[0, 0].set_title('Distribution Fits')
            axes[0, 0].legend()
            
            # Plot 2: Open Interest by Strike
            call_data = self.option_chain[self.option_chain['type'] == 'Call']
            put_data = self.option_chain[self.option_chain['type'] == 'Put']
            
            axes[0, 1].bar(call_data['strike'], call_data['openinterest'],
                          alpha=0.5, color='green', label='Calls')
            axes[0, 1].bar(put_data['strike'], put_data['openinterest'],
                          alpha=0.5, color='red', label='Puts')
            axes[0, 1].set_title('Open Interest Distribution')
            axes[0, 1].legend()
            
            # Plot 3: KDE
            xx, yy, z = self.calculate_kde()
            im = axes[1, 0].contourf(yy, xx, z, cmap='viridis')
            axes[1, 0].set_title('Kernel Density Estimation')
            plt.colorbar(im, ax=axes[1, 0], label='Density')
            
            # Plot 4: Volume Profile
            axes[1, 1].hist(self.option_chain['strike'],
                          weights=self.option_chain['volume'],
                          bins=50, orientation='horizontal')
            axes[1, 1].set_title('Volume Profile')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Plot saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Plot generation failed: {e}")
            raise

    def get_strike_probability(self, strike: float, distribution: str = 'kde') -> float:
        """
        Calculate the probability density at a specific strike price.

        Args:
            strike (float): Strike price to evaluate
            distribution (str): Distribution to use ('kde', 'normal', 'gamma', 'lognormal')

        Returns:
            float: Probability density at the strike price

        Raises:
            ValueError: If invalid distribution is specified
        """
        logger.info(f"Calculating probability for strike {strike} using {distribution}")
        
        try:
            if distribution == 'kde':
                kde = KernelDensity(bandwidth=0.8)
                kde.fit(self.strikes.reshape(-1, 1))
                return np.exp(kde.score_samples([[strike]]))[0]
            
            distributions = self.fit_distributions()
            if distribution not in distributions:
                raise ValueError(f"Invalid distribution: {distribution}")
                
            result = distributions[distribution]
            idx = np.abs(result.strike_prices - strike).argmin()
            return result.pdf_values[idx]
            
        except Exception as e:
            logger.error(f"Probability calculation failed: {e}")
            raise
    
    @staticmethod
    def heston_charfunc(phi: complex, 
                        S0: float,
                        v0: float,
                        kappa: float,
                        theta: float,
                        sigma: float,
                        rho: float,
                        lambd: float,
                        tau: float,
                        r: float
            ) -> float:
        """ 
        Heston model characteristic function.
        
        Args:
            phi: Complex number
            S0: Initial stock price
            v0: Initial Variance under risk-neutral measure
            kappa: Mean reversion rate
            theta: Long-term variance
            sigma: volatility of volatility
            rho: correlation between returns and variances under risk-neutral dynamics
            lambd: risk premium of variance 
            tau: time to maturity
            r: risk-free rate
            
        Returns:
            float: Characteristic function value

        """

        # constants
        a = kappa*theta
        b = kappa+lambd

        # common terms w.r.t phi
        rspi = rho*sigma*phi*1j

        # define d parameter given phi and b
        d = np.sqrt( (rspi - b)**2 + (phi*1j+phi**2)*sigma**2 )

        # define g parameter given phi, b and d
        g = (b-rspi+d)/(b-rspi-d)

        # calculate characteristic function by components
        exp1 = np.exp(r*phi*1j*tau)
        term2 = S0**(phi*1j) * ( (1-g*np.exp(d*tau))/(1-g) )**(-2*a/sigma**2)
        exp2 = np.exp(a*tau*(b-rspi+d)/sigma**2 + v0*(b-rspi+d)*( (1-np.exp(d*tau))/(1-g*np.exp(d*tau)) )/sigma**2)

        return exp1*term2*exp2

    @staticmethod
    # def heston_price_rec(S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r):
    def heston_price_rec(
        S0: float,
        K: float,
        v0: float,
        kappa: float,
        theta: float,
        sigma: float,
        rho: float,
        lambd: float,
        tau: float,
        r: float
        ) -> float:
        """
        Heston model option pricing using rectangular integration method.
        
        Args:
            S0: Initial stock price
            K: Strike price
            v0: Initial Variance under risk-neutral measure
            kappa: Mean reversion rate
            theta: Long-term variance
            sigma: volatility of volatility
            rho: correlation between returns and variances under risk-neutral dynamics
            lambd: risk premium of variance
            tau: time to maturity
            r: risk-free rate
        
        Returns:
            float: Option price
        """
        args = (S0, v0, kappa, theta, sigma, rho, lambd, tau, r)

        P, umax, N = 0, 100, 650
        dphi=umax/N #dphi is width
        for j in range(1,N):
            # rectangular integration
            phi = dphi * (2*j + 1)/2 # midpoint to calculate height
            numerator = heston_charfunc(phi-1j,*args) - K * heston_charfunc(phi,*args)
            denominator = 1j*phi*K**(1j*phi)

            P += dphi * numerator/denominator

        return np.real((S0 - K*np.exp(-r*tau))/2 + P/np.pi)

    @staticmethod
    def heston_price_quad(S0: float,
                          K: float,
                          v0: float,
                          kappa: float,
                          theta: float,
                          sigma: float,
                          rho: float,
                          lambd: float,
                          tau: float,
                          r: float
                          ) -> float:
        """
        Heston model option pricing using quadrature integration method.
        
        Args:
            S0: Initial stock price
            K: Strike price
            v0: Initial Variance under risk-neutral measure
            kappa: Mean reversion rate
            theta: Long-term variance
            sigma: volatility of volatility
            rho: correlation between returns and variances under risk-neutral dynamics
            lambd: risk premium of variance
            tau: time to maturity
            r: risk-free rate
        
        Returns:
            float: Option price
        """
        args = (S0, v0, kappa, theta, sigma, rho, lambd, tau, r)

        integrand = lambda phi: (np.exp(-1j*phi*np.log(K)) * heston_charfunc(phi,*args)/(1j*phi)).real
        integral = quad(integrand, 1e-15, 500, limit=250)[0]

        return (S0 - K*np.exp(-r*tau)/2 + integral/np.pi)
    
    @staticmethod
    def get_price_curvature(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the curvature of option prices.
        
        Args:
            df (pd.DataFrame): DataFrame containing option prices, and strikes. 
                Must have columns 'lastprice' and 'strike'.
        
        Returns:
            pd.DataFrame: DataFrame with curvature values for each strike and option price
        """
        df = 
        # Calculate curvature
        df['curvature'] = (
            -2 * df['lastprice'] +
            df['lastprice'].shift(-1) +
            df['lastprice'].shift(1)
        ) / 1**2
        
        return df
