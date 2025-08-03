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

def calculate_rsi(series, period=14):
    """
    Calculate RSI manually using pandas.
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

# def run_trading_simulation(file_path='../dataHistory/TrainingData/publicData_AAVE-USDT_history_2025-07-23_15_32_12.json'):
def run_trading_simulation(file_path='../dataHistory/TrainingData/publicData_AAVE-USDT_history_2025-08-03_13_18_11.json'):
    try:
        # Load data
        data = load_json_data(file_path)
        timestamps, opens, highs, lows, closes, volumes = data
        print(f"Loaded data shape: {data.shape}")
        print(f"Sample data (first row): {data[:, 0]}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

    # Create 1-min DataFrame
    try:
        df_1min = pd.DataFrame({
            'timestamp': timestamps,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })
        df_1min['datetime'] = pd.to_datetime(df_1min['timestamp'], unit='ms')
        df_1min.set_index('datetime', inplace=True)
        print(f"1-min DataFrame head:\n{df_1min.head()}")
    except Exception as e:
        print(f"Error creating 1-min DataFrame: {e}")
        return None, None

    # Resample to 15-min
    try:
        df_15min = df_1min.resample('15min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        print(f"15-min DataFrame head:\n{df_15min.head()}")
    except Exception as e:
        print(f"Error resampling to 15-min: {e}")
        return None, None
    
    # Bollinger Bands parameters (on 15-min)
    bb_window = 20
    std_mult = 2.0
    
    # Trend parameters (on 15-min)
    trend_window = 10
    adjustment_factor = 59.0
    
    # RSI parameters (on 15-min)
    rsi_window = 14
    rsi_buy_th = 35
    rsi_sell_th = 65
    
    # Volume confirmation
    use_volume_confirmation = False

    # Fees
    buy_sell_fee = 0.001
    
    try:
        df_15min['volume_sma'] = df_15min['volume'].rolling(bb_window).mean()
        df_15min['sma'] = df_15min['close'].rolling(bb_window).mean()
        df_15min['std'] = df_15min['close'].rolling(bb_window).std()
        df_15min['upper'] = df_15min['sma'] + std_mult * df_15min['std']
        df_15min['lower'] = df_15min['sma'] - std_mult * df_15min['std']
        df_15min['percent_b'] = (df_15min['close'] - df_15min['lower']) / (df_15min['upper'] - df_15min['lower'])
        df_15min['sma_prev'] = df_15min['sma'].shift(trend_window)
        df_15min['slope'] = ((df_15min['sma'] - df_15min['sma_prev']) / df_15min['sma']) / trend_window
        df_15min['buy_th'] = 0.0 + adjustment_factor * df_15min['slope']
        df_15min['sell_th'] = 1.0 + adjustment_factor * df_15min['slope']
        
        # RSI
        df_15min['rsi'] = calculate_rsi(df_15min['close'], rsi_window)
        
        df_15min['buy_th_price'] = df_15min['lower'] + (df_15min['upper'] - df_15min['lower']) * df_15min['buy_th']
        df_15min['sell_th_price'] = df_15min['lower'] + (df_15min['upper'] - df_15min['lower']) * df_15min['sell_th']
    except Exception as e:
        print(f"Error computing indicators on 15-min: {e}")
        return None, None
    
    # Map 15-min indicators to 1-min
    df_1min = df_1min.join(df_15min[['sma', 'upper', 'lower', 'percent_b', 'buy_th_price', 'sell_th_price', 'volume_sma', 'rsi']], how='left')
    df_1min[['sma', 'upper', 'lower', 'percent_b', 'buy_th_price', 'sell_th_price', 'volume_sma', 'rsi']] = df_1min[['sma', 'upper', 'lower', 'percent_b', 'buy_th_price', 'sell_th_price', 'volume_sma', 'rsi']].ffill()
    
    # Backtest variables
    initial_cash = 10000.0
    cash = initial_cash
    position = 0.0
    entry_price = 0.0
    trades = []
    equity = []
    buy_times = []
    buy_prices = []
    sell_times = []
    sell_prices = []
    
    # Debug data
    debug_data = []
    
    # Loop through 1-min
    for i in range(len(df_1min)):
        if i == 0:
            equity.append(cash if position == 0 else position * df_1min.iloc[i]['close'])
        
        row_1min = df_1min.iloc[i]
        price = row_1min['close']
        low_price = row_1min['low']
        ts = row_1min.name
        
        if pd.isna(row_1min['percent_b']) or pd.isna(row_1min['rsi']):
            equity.append(equity[-1])
            continue
        
        try:
            if position > 0:
                if low_price < entry_price * 0.98:
                    cash = position * price * (1 - buy_sell_fee)
                    trades.append(f"Minute {i} (timestamp {row_1min['timestamp']}): STOP-LOSS SELL at {price:.2f}, Cash: {cash:.2f}")
                    sell_times.append(row_1min['timestamp'])
                    sell_prices.append(price)
                    position = 0.0
                    equity.append(cash)
                    continue
            
            if i % 15 == 14:
                bin_label = ts.floor('15min')
                if bin_label in df_15min.index:
                    row_15min = df_15min.loc[bin_label]
                else:
                    equity.append(equity[-1])
                    continue
                
                percent_b = row_15min['percent_b']
                buy_th = row_15min['buy_th']
                sell_th = row_15min['sell_th']
                rsi = row_15min['rsi']
                volume = row_15min['volume']
                volume_sma = row_15min['volume_sma']
                volume_ok = (not use_volume_confirmation) or (volume > volume_sma)
                
                # Debug
                debug_data.append({
                    'minute': i,
                    'timestamp': row_1min['timestamp'],
                    'close': price,
                    'percent_b': percent_b,
                    'buy_th': buy_th,
                    'sell_th': sell_th,
                    'rsi': rsi,
                    'volume': volume,
                    'volume_sma': volume_sma if not pd.isna(volume_sma) else np.nan,
                    'volume_ok': volume_ok,
                    'in_position': position > 0
                })
                
                # Log missed buys
                if position == 0 and percent_b < buy_th and rsi >= rsi_buy_th:
                    print(f"Missed buy at minute {i} (ts {row_1min['timestamp']}): %B={percent_b:.4f} < buy_th={buy_th:.4f}, but RSI={rsi:.2f} >= {rsi_buy_th}")
                elif position == 0 and percent_b < buy_th and not volume_ok:
                    print(f"Missed buy at minute {i} (ts {row_1min['timestamp']}): %B={percent_b:.4f} < buy_th={buy_th:.4f}, but volume={volume:.2f} <= volume_sma={volume_sma:.2f}")
                
                if position == 0 and percent_b < buy_th and rsi < rsi_buy_th and volume_ok:
                    position = cash / price * (1 - buy_sell_fee)
                    entry_price = price
                    cash = 0.0
                    trades.append(f"Minute {i} (timestamp {row_1min['timestamp']}): BUY at {price:.2f}, Position: {position:.4f}")
                    buy_times.append(row_1min['timestamp'])
                    buy_prices.append(price)
                
                elif position > 0 and percent_b > sell_th and rsi > rsi_sell_th:
                    cash = position * price * (1 - buy_sell_fee)
                    position = 0.0
                    trades.append(f"Minute {i} (timestamp {row_1min['timestamp']}): SELL at {price:.2f}, Cash: {cash:.2f}")
                    sell_times.append(row_1min['timestamp'])
                    sell_prices.append(price)
            
            equity.append(position * price if position > 0 else cash)
        except Exception as e:
            print(f"Error in trading loop at minute {i}: {e}")
            continue
    
    # Close open position
    if position > 0:
        final_price = df_1min['close'].iloc[-1]
        cash = position * final_price * (1 - buy_sell_fee)
        trades.append(f"Final: CLOSE POSITION at {final_price:.2f}, Cash: {cash:.2f}")
        sell_times.append(df_1min.iloc[-1]['timestamp'])
        sell_prices.append(final_price)
        equity[-1] = cash
    
    # Fix equity length
    if len(equity) > len(df_1min):
        equity = equity[:len(df_1min)]
    elif len(equity) < len(df_1min):
        equity.extend([equity[-1]] * (len(df_1min) - len(equity)))
    
    print(f"df_1min length: {len(df_1min)}")
    print(f"equity length: {len(equity)}")
    
    # Save debug data
    try:
        debug_df = pd.DataFrame(debug_data)
        debug_df.to_csv('trade_debug.csv', index=False)
        print("Debug data saved to 'trade_debug.csv'")
    except Exception as e:
        print(f"Error saving debug data: {e}")
    
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
        
        # Plot
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 12), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1, 1]})
        
        ax1.plot(df_1min['timestamp'], df_1min['close'], label='Price (Close)', color='blue', linewidth=1)
        ax1.plot(df_1min['timestamp'], df_1min['sma'], label='SMA', color='orange', linewidth=1)
        ax1.plot(df_1min['timestamp'], df_1min['upper'], label='Upper Band', color='green', linestyle='--')
        ax1.plot(df_1min['timestamp'], df_1min['lower'], label='Lower Band', color='red', linestyle='--')
        ax1.plot(df_1min['timestamp'], df_1min['buy_th_price'], label='Buy Threshold', color='lime', linestyle=':')
        ax1.plot(df_1min['timestamp'], df_1min['sell_th_price'], label='Sell Threshold', color='magenta', linestyle=':')
        ax1.scatter(buy_times, buy_prices, color='green', marker='o', s=50, label='Buy')
        ax1.scatter(sell_times, sell_prices, color='red', marker='o', s=50, label='Sell/Stop-Loss')
        ax1.set_title('Price with Bollinger Bands and Thresholds (15-min)')
        ax1.set_ylabel('Price (USDT)')
        ax1.legend(loc='upper left', fontsize='small')
        ax1.grid(True)
        
        ax2.plot(df_1min['timestamp'], equity, label='Equity', color='purple')
        ax2.set_title('Equity Curve')
        ax2.set_ylabel('Equity')
        ax2.legend()
        ax2.grid(True)
        
        ax3.plot(df_1min['timestamp'], df_1min['volume'], label='Volume (1-min)', color='gray')
        ax3.plot(df_1min['timestamp'], df_1min['volume_sma'], label='Volume SMA', color='brown', linestyle='--')
        ax3.set_title('Volume')
        ax3.set_ylabel('Volume')
        ax3.legend()
        ax3.grid(True)
        
        ax4.plot(df_1min['timestamp'], df_1min['rsi'], label='RSI (15-min)', color='cyan')
        ax4.axhline(y=rsi_buy_th, color='lime', linestyle='--', label='RSI Buy Th')
        ax4.axhline(y=rsi_sell_th, color='magenta', linestyle='--', label='RSI Sell Th')
        ax4.set_title('RSI')
        ax4.set_xlabel('Timestamp')
        ax4.set_ylabel('RSI')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error displaying results or plotting: {e}")
    
    return df_1min, equity

# Run the simulation
df, equity = run_trading_simulation()