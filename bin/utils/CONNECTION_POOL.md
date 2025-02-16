# Database Connection Pool

## Overview

The connection pool provides efficient management of SQLite database connections with the following features:

- Thread-safe connection pooling
- Automatic connection lifecycle management
- Connection reuse to reduce overhead
- Monitoring and status reporting
- Support for read-only connections
- Automatic cleanup of idle/expired connections

## Integration Guide

### Basic Usage

```python
from utils.connection_pool import get_pool

# Get the global connection pool instance
pool = get_pool()

# Use a connection from the pool
with pool.get_connection('daily') as conn:
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM some_table')
    results = cursor.fetchall()
```

### Migrating Existing Code

#### From Current Implementation:
```python
class Prices:
    def __init__(self, connections):
        self.daily_db = sql.connect(connections['daily_db'])
        self.intraday_db = sql.connect(connections['intraday_db'])
        
    def custom_q(self, q: str):
        cursor = self.daily_db.cursor()
        cursor.execute(q)
        return cursor.fetchall()
```

#### To Connection Pool:
```python
from utils.connection_pool import get_pool

class Prices:
    def __init__(self, connections):
        self.pool = get_pool()
        
    def custom_q(self, q: str):
        with self.pool.get_connection('daily') as conn:
            cursor = conn.cursor()
            cursor.execute(q)
            return cursor.fetchall()
```

### Available Database Pools

The connection pool maintains connections for:

- `daily` - Daily stock prices database
- `intraday` - Intraday stock prices database
- `options` - Options database (write enabled)
- `options_ro` - Options database (read-only)
- `changes` - Options changes database
- `volume` - Options volume database
- `stats` - Options statistics database
- `tracking` - Tracking database
- `tracking_values` - Tracking values database

### Thread Safety

The connection pool is thread-safe and handles concurrent access properly:

```python
import threading

def worker():
    pool = get_pool()
    with pool.get_connection('daily') as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT 1')

# Create multiple threads
threads = [threading.Thread(target=worker) for _ in range(5)]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
```

### Monitoring

Monitor pool status and connection usage:

```python
pool = get_pool()
status = pool.get_pool_status()

for db_name, stats in status.items():
    print(f"\n{db_name}:")
    print(f"  Available connections: {stats['available']}")
    print(f"  In-use connections: {stats['in_use']}")
    print(f"  Total connections: {stats['total']}")
    print(f"  Max connections: {stats['max_connections']}")
```

### Configuration

The connection pool is configured through environment variables in `.env`:

```ini
DAILY_DB="data/prices/stocks.db"
INTRADAY_DB="data/prices/stocks_intraday.db"
OPTION_DB="data/options/options.db"
# ... other database paths
```

Pool settings can be customized when creating a new pool:

```python
from utils.connection_pool import ConnectionPool

custom_pool = ConnectionPool(
    min_connections=5,    # Minimum connections to maintain
    max_connections=20    # Maximum connections allowed
)
```

### Error Handling

The connection pool handles errors gracefully:

```python
from utils.connection_pool import get_pool
import sqlite3

pool = get_pool()

try:
    with pool.get_connection('daily') as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM nonexistent_table')
except sqlite3.Error as e:
    print(f"Database error: {str(e)}")
```

### Connection Lifecycle

Connections are automatically managed:

- New connections are created when needed (up to max_connections)
- Idle connections are cleaned up after max_idle_time (default: 5 minutes)
- Connections are retired after max_lifetime (default: 1 hour)
- Active connections are monitored and force-closed if expired
- All resources are properly cleaned up when the pool is closed

### Best Practices

1. Always use the context manager (`with` statement) to ensure proper connection return
2. Use read-only connections when possible (`options_ro` instead of `options`)
3. Keep operations within connection context as short as possible
4. Monitor pool status in production environments
5. Handle SQLite errors appropriately in your code
6. Close the pool when shutting down your application

### Logging

The connection pool logs important events to `logs/connection_pool.log`:

- Connection creation/closure
- Pool initialization
- Error conditions
- Maintenance activities
- Force-closure of expired connections

### Testing

Comprehensive test suite available in `test_connection_pool.py`:

```bash
# Run all tests
python -m unittest bin/utils/test_connection_pool.py -v

# Run specific test
python -m unittest bin.utils.test_connection_pool.TestConnectionPool.test_concurrent_access