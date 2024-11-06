
## Stock Tracking App
![alt text](setup/image.png)

## Description
This is a stock tracking app that allows users to track prices, and options for various stocks using the Yfinance API. The app does not require an API key, and is free to use. 


## Table of Contents
- [Stock Tracking App](#stock-tracking-app)
- [Description](#description)
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
- [Usage](#usage)
- [Price Module](#price-module)
- [Options Module](#options-module)
- [Contributing](#contributing)
- [License](#license)


## Installation
To install necessary dependencies, install the packages in the setup/requirements.txt file. You can do this by running the following command in your terminal:

```
pip install -r setup/requirements.txt
```

After installing the packages run the main.py file to initialize the app. You can do this from a terminal by running the following command in the root directory of the app: 
    
    ```
    python3 main.py 
    ```

Initialize a list of stocks by saving them to a text file in the setup/ directory, the files should be named 'stocks.txt'. Each stock should be on a new line. You can also add stocks at any time by utilizing the `.addStocks(stock)` method in the main.py file. 


## Usage
Each time the workflow is ran, the app updates the database with Intraday, and Daily price data along with the current Option Chain for each stock. 


## Price Module 
Used for updating price data, retrieving price data, and calculating technical indicators.
  - Contains 3 Main Modules: 
    - `update_stocks`:
      - Updates the database with the latest Intraday and Daily price data
    - `Prices`:
      - Contains methods for retrieving price data from the database, Methods include: 
        - `.get_intraday_close(stocks, interval)`
          - Gets the intraday closing price for a stock over a specified interval, the interval can be in minutes (T), hours (H)
        - `.get_close(stocks, startdate, end_date)`
          - Gets the closing price for a stock over a specified date range
        - `.ohlc(stock, daily_data, start_date, end_date)`
          - Returns the Open, High, Low, Close, and Volume for a stock over a specified date range
        - `.sectors()`
          - Returns the Sector performance for stocks included in the database.
        - `.industries()`
          - Returns the Industry performance for stocks included in the database. 
        - `.daily_aggregates(stock)`
          - Returns daily aggregated price points, including Daily, Weekly, and Monthly data 
        - `.intra_day_aggs(price_df)`
          - Returns Intraday aggregated price points, including 3, 6, 18 minute, 1 hour, and 4 hour aggregated data. 
        - `.get_indicators()`
          - Returns Technical indicators using the Indicator module below. 

    - `Indicators` 
      - Moving Averages, EMA, SMA, KAMA, and MACD
      - Volatility Indicators: ATR, Bollinger Bands, Keltner Channels
      - Momentum Indicators: RSI, Momentum, Stochastic Oscillator 
      - Mean Reversion Indicaotr
      - Price Levels of interest
      - allows for quick translation of indicator data into buying and selling signals, using the `get_states()` method.
      - This method is still in development, and will be updated in the future, if you would like to contribute to this project, please see the [Contributing](#contributing) section.


## Options Module

Contains 3 main Modules: 
  -  OptionChain
     -  The OptionsChain Module is responsible for updating the database with the latest Option Chain data 
     -  Some methods that are frequently used: 
     -  `today_option_chain(stock, bsdf=True)`
        -  Returns the most recent Option Chain, and gives the option to return the data with Option Greeks Calculated from the Black-scholes Model. 
     -  `today_option_chain_cdb(stock, bsdf=True)`
        -  Returns the most recent Option Chain along with the observed changes from the previous data pull. <br></br>
  -  Stats
     -  The Stats Module is responsible for Aggregating and Analyzing Option Chain Data. All methods are currently implemented into the workflow, so after a few data grabs, you will be able to populate the database with the following statistics: <br></br>
        -  `ChangeVars` is responsible for calculating changes in option contracts, given the current and previous option chain data. 
           -  This is then saved to a seperate database, without identifiers like Expiration data, strike price, etc. (For faster querying), The `parse_change_db` function within the OptionChain Module is frequently used with this class, as it allows for translating contractsymbols into the identifiers mentioned above.
        -  `CP` is a class used to track overall option chain statistics for each stock. It tracks the flow of open interest, volume, premium, implied volatility, price spreads along with their respective changes. 
        -  `EXP` is a class used to calculate the expected move of a stock based upon the current Option chain. 
           -  This is done by taking the ratio of the ATM Straddle to the stock price, to get the percentage move.<br></br>
    
  -  Tracking (`Screener`)
     -  This module is responsible for tracking changes in Contracts that have been identified as having Favorabble circumstances, such as a high volume of contracts traded, or a high change in open interest.
     -  Once Contracts are identified, the `Tracker` module is used to track the P/L along changes in IV, Volume, etc, after the contract has been identified. 
     -  This method is still in development, and will be updated in the future, if you would like to contribute to this project, please see the [Contributing](#contributing) section.




## Contributing
If you would like to contribute to this project, please fork the repository, and submit a pull request. I am always looking for ways to improve the app, and would love to hear and implement your ideas.


## License
This project is licensed under the MIT License - see the LICENSE.md file for details






