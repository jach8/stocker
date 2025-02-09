# Option Chain Density Analysis

A robust Python module for analyzing option chain density patterns and probability distributions. This module provides tools for processing open interest and volume data to calculate density metrics per expiration date.

## Features

- Multiple density estimation methods:
  - Parametric distributions (Normal, Gamma, Log-normal)
  - Kernel Density Estimation (KDE)
- Professional-grade visualizations with clear legends and annotations
- Comprehensive error handling and data validation
- Type hints following PEP 484
- Structured logging for execution flow tracking

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

```python
from option_chain_density import OptionChainDensity

# Initialize with option chain data
analyzer = OptionChainDensity(option_chain_df)

# Fit probability distributions
distributions = analyzer.fit_distributions()

# Calculate KDE
xx, yy, z = analyzer.calculate_kde()

# Generate comprehensive analysis plots
analyzer.plot_density_analysis()

# Calculate probability for specific strike
probability = analyzer.get_strike_probability(strike_price, distribution='kde')
```

## Input Data Format

The OptionChainDensity class expects a pandas DataFrame with the following columns:
- `strike`: Strike prices (float)
- `type`: Option type ('Call' or 'Put')
- `openinterest`: Open interest values (int)
- `volume`: Trading volume (int)
- `expiry`: Expiration dates (datetime-like)

## Key Methods

### fit_distributions()
Fits multiple probability distributions to the option chain data and returns goodness-of-fit metrics.

### calculate_kde(bandwidth=0.8)
Calculates Kernel Density Estimation for more flexible density analysis.

### plot_density_analysis(save_path=None)
Generates comprehensive visualization including:
- Distribution fits
- Open interest distribution
- KDE contour plot
- Volume profile

### get_strike_probability(strike, distribution='kde')
Calculates probability density at a specific strike price using the specified distribution method.

## Error Handling

The module includes comprehensive error handling for:
- Input data validation
- Distribution fitting failures
- KDE calculation issues
- Plotting errors

All errors are logged with detailed messages for debugging.

## Example

See `density_analysis_example.ipynb` for a complete demonstration of the module's capabilities.

## Contributing

Feel free to submit issues and enhancement requests!

## License

MIT