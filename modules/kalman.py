import numpy as np

def apply_kalman(prices):
    # Ensure prices is a valid Pandas Series and not empty
    if prices.empty:
        raise ValueError("The prices series is empty.")

    print(f"Prices Series: {prices.head()}")  # Print first few entries for debugging
    
    # Initialize the Kalman filter parameters
    n = len(prices)
    x = np.zeros(n)
    P = np.zeros(n)
    Q = 1e-5  # Process variance
    R = 0.01  # Estimate of measurement variance

    # Initial guesses
    try:
        x[0] = prices.iloc[0]
    except IndexError:
        raise ValueError("Index error: the prices series may not be properly indexed.")
    
    P[0] = 1.0

    for k in range(1, n):
        # Time update
        x[k] = x[k-1]
        P[k] = P[k-1] + Q

        # Measurement update
        try:
            K = P[k] / (P[k] + R)
            x[k] = x[k] + K * (prices.iloc[k] - x[k])
            P[k] = (1 - K) * P[k]
        except IndexError:
            raise ValueError(f"Index error at position {k}: the prices series may not be properly indexed.")
    
    return x
