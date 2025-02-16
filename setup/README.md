# Database Setup Script

## Overview

This script handles the complete setup and validation of the database infrastructure for the stocker project. It performs the following tasks:

1. Validates database connection strings and credentials from `.env`
2. Creates required directories if they don't exist
3. Tests connectivity to each database
4. Creates missing databases with proper schemas
5. Handles errors gracefully with detailed logging
6. Reports success/failure status for each connection
7. Creates any missing configuration files
8. Sets up proper permissions and access rights
9. Validates JSON configuration files
10. Provides a summary of completed setup actions

## Prerequisites

1. Python 3.8 or higher
2. Required Python packages (install using `pip`):
   ```bash
   pip install -r setup/requirements.txt
   ```

3. Valid `.env` file in the project root with the following configuration:
   ```ini
   # API Keys
   ALPHA_ADVANTAGE_API_KEY="YOUR_API_KEY"

   # Database Paths
   DAILY_DB="data/prices/stocks.db"
   INTRADAY_DB="data/prices/stocks_intraday.db"
   OPTION_DB="data/options/options.db"
   CHANGE_DB="data/options/option_change.db"
   VOL_DB="data/options/vol.db"
   STATS_DB="data/options/stats.db"
   TRACKING_DB="data/options/tracking.db"
   TRACKING_VALUES_DB="data/options/tracking_values.db"
   BACKUP_DB="data/options/log/backup.db"
   INACTIVE_DB="data/options/log/inactive.db"

   # Configuration Files
   TICKER_PATH="data/stocks/tickers.json"
   STOCK_INFO_DICT="data/stocks/stock_info.json"
   ```

## Usage

1. Make sure all prerequisites are installed and `.env` file is configured
2. Run the setup script:
   ```bash
   ./setup/setup_database.py
   ```

The script will:
- Create all necessary directories
- Set up databases with proper schemas
- Validate all connections and permissions
- Generate a detailed setup report
- Create a log file at `logs/setup.log`

## Logging

The script logs all actions to:
- Console (INFO level and above)
- `logs/setup.log` file (DEBUG level and above, rotates at 10MB)

## Error Handling

- The script handles errors gracefully and provides detailed error messages
- If any critical step fails, the script will exit with a non-zero status code
- All errors are logged with stack traces in the log file
- The summary report will show the status of each component

## Directory Structure Created

```
data/
├── prices/
│   ├── stocks.db
│   └── stocks_intraday.db
├── options/
│   ├── options.db
│   ├── option_change.db
│   ├── vol.db
│   ├── stats.db
│   ├── tracking.db
│   ├── tracking_values.db
│   └── log/
│       ├── backup.db
│       └── inactive.db
├── stocks/
│   ├── tickers.json
│   └── stock_info.json
├── bonds/
└── earnings/

logs/
└── setup.log
```

## Troubleshooting

If you encounter any issues:

1. Check the `logs/setup.log` file for detailed error messages
2. Verify all paths in `.env` are correct
3. Ensure you have write permissions in all required directories
4. Validate JSON configuration files manually if needed

## Exit Codes

- 0: Setup completed successfully
- 1: Setup failed (check logs for details)