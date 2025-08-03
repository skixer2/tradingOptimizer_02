import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

def load_json_data(file_path):
    """
    Load OHLCV data from JSON file and convert to tradingData format.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
   
    ohlcv = data['data']
    ohlcv = sorted(ohlcv, key=lambda x: int(x[0]))
   
    timestamps = [int(row[0]) for row in ohlcv]
    opens = [float(row[1]) for row in ohlcv]
    highs = [float(row[2]) for row in ohlcv]
    lows = [float(row[3]) for row in ohlcv]
    closes = [float(row[4]) for row in ohlcv]
    volumes = [float(row[5]) for row in ohlcv]
   
    return np.array([timestamps, opens, highs, lows, closes, volumes])

def run_trading_simulation(file_path='../dataHistory/TrainingData/publicData_AAVE-USDT_history_2025-07-23_15_32_12.json'):
    try:
        # Load data
        data = load_json_data(file_path)
        timestamps, opens, highs, lows, closes, volumes = data
        print(f"Loaded data shape: {data.shape}")
        print(f"Sample data (first row): {data[:, 0]}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

    # Create DataFrame
    try:
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })
        print(f"DataFrame head:\n{df.head()}")
    except Exception as e:
        print(f"Error creating DataFrame: {e}")
        return None, None
    
    # Bollinger Bands parameters
    bb_window = 20
    std_mult = 2.0
    
    # Trend parameters
    trend_window = 10
    adjustment_factor = 50.0
    
    # Volume confirmation
    use_volume_confirmation = True
    try:
        df['volume_sma'] = df['volume'].rolling(bb_window).mean()
        df['sma'] = df['close'].rolling(bb_window).mean()
        df['std'] = df['close'].rolling(bb_window).std()
        df['upper'] = df['sma'] + std_mult * df['std']
        df['lower'] = df['sma'] - std_mult * df['std']
        df['percent_b'] = (df['close'] - df['lower']) / (df['upper'] - df['lower'])
        df['sma_prev'] = df['sma'].shift(trend_window)
        df['slope'] = ((df['sma'] - df['sma_prev']) / df['sma']) / trend_window
        df['buy_th'] = 0.0 + adjustment_factor * df['slope']
        df['sell_th'] = 1.0 + adjustment_factor * df['slope']
        
        # Convert thresholds to price levels for plotting
        df['buy_th_price'] = df['lower'] + (df['upper'] - df['lower']) * df['buy_th']
        df['sell_th_price'] = df['lower'] + (df['upper'] - df['lower']) * df['sell_th']
    except Exception as e:
        print(f"Error computing indicators: {e}")
        return None, None
    
    # Backtest variables
    initial_cash = 10000.0
    cash = initial_cash
    position = 0.0
    entry_price = 0.0
    trades = []
    equity = []  # Initialize empty
    
    # Lists for plotting buy/sell markers
    buy_times = []
    buy_prices = []
    sell_times = []
    sell_prices = []
    
    # Loop through every minute
    for i in range(len(df)):
        # Set initial equity on first iteration
        if i == 0:
            equity.append(cash if position == 0 else position * df.loc[i, 'close'])
        
        if pd.isna(df.loc[i, 'percent_b']) or pd.isna(df.loc[i, 'slope']):
            equity.append(equity[-1])
            continue
        
        price = df.loc[i, 'close']
        low_price = df.loc[i, 'low']
        
        try:
            # Stop-loss check
            if position > 0:
                if low_price < entry_price * 0.98:
                    cash = position * price
                    trades.append(f"Minute {i} (timestamp {df.loc[i, 'timestamp']}): STOP-LOSS SELL at {price:.2f}, Cash: {cash:.2f}")
                    sell_times.append(df.loc[i, 'timestamp'])
                    sell_prices.append(price)
                    position = 0.0
                    equity.append(cash)
                    continue
            
            # Trading decisions every 15 minutes
            if i % 15 == 0:
                percent_b = df.loc[i, 'percent_b']
                buy_th = df.loc[i, 'buy_th']
                sell_th = df.loc[i, 'sell_th']
                volume_ok = (not use_volume_confirmation) or (df.loc[i, 'volume'] > df.loc[i, 'volume_sma'])
                
                if position == 0 and percent_b < buy_th and volume_ok:
                    position = cash / price
                    entry_price = price
                    cash = 0.0
                    trades.append(f"Minute {i} (timestamp {df.loc[i, 'timestamp']}): BUY at {price:.2f}, Position: {position:.4f}")
                    buy_times.append(df.loc[i, 'timestamp'])
                    buy_prices.append(price)
                
                elif position > 0 and percent_b > sell_th:
                    cash = position * price
                    position = 0.0
                    trades.append(f"Minute {i} (timestamp {df.loc[i, 'timestamp']}): SELL at {price:.2f}, Cash: {cash:.2f}")
                    sell_times.append(df.loc[i, 'timestamp'])
                    sell_prices.append(price)
            
            # Update equity
            equity.append(position * price if position > 0 else cash)
        except Exception as e:
            print(f"Error in trading loop at minute {i}: {e}")
            continue
    
    # Close any open position
    if position > 0:
        try:
            final_price = df['close'].iloc[-1]
            cash = position * final_price
            trades.append(f"Final: CLOSE POSITION at {final_price:.2f}, Cash: {cash:.2f}")
            sell_times.append(df['timestamp'].iloc[-1])
            sell_prices.append(final_price)
            equity[-1] = cash  # Update last equity
        except Exception as e:
            print(f"Error closing position: {e}")
    
    # Ensure equity length matches df
    if len(equity) > len(df):
        equity = equity[:len(df)]
    elif len(equity) < len(df):
        equity.extend([equity[-1]] * (len(df) - len(equity)))
    
    print(f"df['timestamp'] length: {len(df['timestamp'])}")
    print(f"equity length: {len(equity)}")
    
    # Results
    try:
        final_value = cash
        profit = final_value - initial_cash
        return_pct = (profit / initial_cash) * 100
        
        print("\nTrade Log:")
        for trade in trades:
            print(trade)
        
        print(f"\nInitial Cash: {initial_cash:.2f}")
        print(f"Final Value: {final_value:.2f}")
        print(f"Total Profit: {profit:.2f}")
        print(f"Return: {return_pct:.2f}%")
        
        # Enhanced Plot: Two subplots - Price with BB, thresholds, markers; Equity curve
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        
        # Top plot: Price evolution, Bollinger Bands, thresholds, buy/sell markers
        ax1.plot(df['timestamp'], df['close'], label='Price (Close)', color='blue', linewidth=1)
        ax1.plot(df['timestamp'], df['sma'], label='SMA (Middle Band)', color='orange', linewidth=1)
        ax1.plot(df['timestamp'], df['upper'], label='Upper Band', color='green', linestyle='--')
        ax1.plot(df['timestamp'], df['lower'], label='Lower Band', color='red', linestyle='--')
        ax1.plot(df['timestamp'], df['buy_th_price'], label='Buy Threshold (Price)', color='lime', linestyle=':', alpha=0.7)
        ax1.plot(df['timestamp'], df['sell_th_price'], label='Sell Threshold (Price)', color='magenta', linestyle=':', alpha=0.7)
        
        # Buy/sell markers (small circles)
        ax1.scatter(buy_times, buy_prices, color='green', marker='o', s=50, label='Buy')
        ax1.scatter(sell_times, sell_prices, color='red', marker='o', s=50, label='Sell/Stop-Loss')
        
        ax1.set_title('Price Evolution with Bollinger Bands, Thresholds, and Trades')
        ax1.set_ylabel('Price (USDT)')
        ax1.legend(loc='upper left', fontsize='small')
        ax1.grid(True)
        
        # Bottom plot: Equity curve
        ax2.plot(df['timestamp'], equity, label='Equity (USDT)', color='purple')
        ax2.set_title('Equity Curve')
        ax2.set_xlabel('Timestamp')
        ax2.set_ylabel('Equity')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error displaying results or plotting: {e}")
    
    return df, equity

# Run the simulation
df, equity = run_trading_simulation()