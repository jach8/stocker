from connection_pool import get_pool
if __name__ == "__main__":
    # Example usage
    pool = get_pool()
    
    try:
        # Get connection from pool
        with pool.get_connection('daily') as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT 1')
            result = cursor.fetchone()
            print(f"Test query result: {result}")
            
        # Print pool status
        status = pool.get_pool_status()
        print("\nPool Status:")
        for db_name, stats in status.items():
            print(f"\n{db_name}:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
                
    finally:
        pool.close_all()