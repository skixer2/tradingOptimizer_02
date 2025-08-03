import json
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

def load_json_data(file_path):
    """
    Load OHLCV data from JSON file and convert to tradingData format.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    ohlcv = data['data']
    ohlcv = sorted(ohlcv, key=lambda x: int(x[0]))  # Sort ascending by timestamp
    
    timestamps = [int(row[0]) for row in ohlcv]
    opens = [float(row[1]) for row in ohlcv]
    highs = [float(row[2]) for row in ohlcv]
    lows = [float(row[3]) for row in ohlcv]
    closes = [float(row[4]) for row in ohlcv]
    volumes = [float(row[5]) for row in ohlcv]
    
    return np.array([timestamps, opens, highs, lows, closes, volumes])

# def backtest_strategy(tradingData, initial_cash=10000, bollinger_period=20, bollinger_std=2,
#                      macd_fast=12, macd_slow=26, macd_signal=9, rsi_period=14,
#                      bb_low_threshold=0.0, bb_high_threshold=1.0,
#                      stop_loss_pct=-0.01, take_profit_pct=0.02, trading_fee=0.0005,
#                      stoch_k_period=14, stoch_d_period=5, stoch_smooth=3,
#                      atr_period=20):
def backtest_strategy(tradingData, initial_cash=10000, bollinger_period=20, bollinger_std=2,
                     macd_fast=12, macd_slow=26, macd_signal=9, rsi_period=14,
                     bb_low_threshold=0.0, bb_high_threshold=1.0,
                     stop_loss_pct=-0.01, take_profit_pct=0.02, trading_fee=0.001,
                     stoch_k_period=14, stoch_d_period=5, stoch_smooth=3,
                     atr_period=20):
    """
    Backtest the trading strategy with relaxed conditions and debug logging.
    """
    df = pd.DataFrame({
        'timestamp': tradingData[0],
        'open': tradingData[1],
        'high': tradingData[2],
        'low': tradingData[3],
        'close': tradingData[4],
        'volume': tradingData[5]
    })
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    # Compute Bollinger Bands
    df['sma'] = df['close'].rolling(bollinger_period).mean()
    df['std'] = df['close'].rolling(bollinger_period).std()
    df['upper_bb'] = df['sma'] + bollinger_std * df['std']
    df['lower_bb'] = df['sma'] - bollinger_std * df['std']
    df['percent_b'] = (df['close'] - df['lower_bb']) / (df['upper_bb'] - df['lower_bb'])

    # Compute MACD
    def ema(series, period):
        return series.ewm(span=period, adjust=False).mean()
    ema_fast = ema(df['close'], macd_fast)
    ema_slow = ema(df['close'], macd_slow)
    df['macd_line'] = ema_fast - ema_slow
    df['macd_signal'] = ema(df['macd_line'], macd_signal)
    df['macd_hist'] = df['macd_line'] - df['macd_signal']

    # Compute RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(rsi_period).mean()
    loss = -delta.where(delta < 0, 0).rolling(rsi_period).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Compute Stochastic Oscillator
    df['lowest_low'] = df['low'].rolling(stoch_k_period).min()
    df['highest_high'] = df['high'].rolling(stoch_k_period).max()
    df['stoch_k'] = 100 * (df['close'] - df['lowest_low']) / (df['highest_high'] - df['lowest_low'])
    df['stoch_d'] = df['stoch_k'].rolling(stoch_d_period).mean()
    df['stoch_smooth'] = df['stoch_d'].rolling(stoch_smooth).mean()

    # Compute Keltner Channels
    df['ema_keltner'] = df['close'].ewm(span=atr_period, adjust=False).mean()
    df['atr'] = df['high'].rolling(atr_period).max() - df['low'].rolling(atr_period).min()
    df['upper_keltner'] = df['ema_keltner'] + (df['atr'] * 2)
    df['lower_keltner'] = df['ema_keltner'] - (df['atr'] * 2)
    df['middle_keltner'] = df['ema_keltner']

    # Relaxed volatility filter
    df['atr_median'] = df['atr'].rolling(50).median()
    df['low_volatility'] = df['atr'] < (df['atr_median'] * 1.5)  # Looser threshold

    # Drop NaN rows
    min_periods = max(bollinger_period, macd_slow, rsi_period, stoch_k_period, atr_period, 50)
    df = df.iloc[min_periods - 1:]

    cash = initial_cash
    position = 0
    buy_price = 0
    in_cooldown = False
    trades = []
    equity = [initial_cash] * len(df)

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]

        # Debug logging
        if prev_row['percent_b'] >= bb_low_threshold and row['percent_b'] < bb_low_threshold:
            print(f"{row.name}: BB Buy Signal, Stoch_K={row['stoch_k']:.2f}, Close={row['close']:.2f}, Lower_Keltner={row['lower_keltner']:.2f}, Low_Vol={row['low_volatility']}")
        if position > 0 and prev_row['percent_b'] <= bb_high_threshold and row['percent_b'] > bb_high_threshold:
            print(f"{row.name}: BB Sell Signal, Stoch_K={row['stoch_k']:.2f}, Close={row['close']:.2f}, Middle_Keltner={row['middle_keltner']:.2f}")

        # Update equity
        equity[i] = cash + (position * row['close'])

        # Stop loss or take profit
        if position > 0:
            profit_pct = (row['close'] - buy_price) / buy_price
            if profit_pct < stop_loss_pct or profit_pct >= take_profit_pct:
                sell_amount = position * row['close'] * (1 - trading_fee)
                cash += sell_amount
                trades.append({
                    'time': row.name,
                    'action': 'sell_stop' if profit_pct < stop_loss_pct else 'sell_tp',
                    'price': row['close'],
                    'position': position,
                    'cash_after': cash,
                    'profit_pct': profit_pct,
                    'fee': position * row['close'] * trading_fee
                })
                position = 0
                in_cooldown = True
                equity[i] = cash
                continue

        # Sell condition
        if position > 0:
            if (prev_row['percent_b'] <= bb_high_threshold and row['percent_b'] > bb_high_threshold and prev_row['stoch_k'] > 70) or row['close'] >= row['middle_keltner']:
                sell_amount = position * row['close'] * (1 - trading_fee)
                cash += sell_amount
                profit_pct = (row['close'] - buy_price) / buy_price
                trades.append({
                    'time': row.name,
                    'action': 'sell',
                    'price': row['close'],
                    'position': position,
                    'cash_after': cash,
                    'profit_pct': profit_pct,
                    'fee': position * row['close'] * trading_fee
                })
                position = 0
                equity[i] = cash

        # Buy condition
        if position == 0 and not in_cooldown:
            if (prev_row['percent_b'] >= bb_low_threshold and row['percent_b'] < bb_low_threshold and
                row['stoch_k'] < 30 and row['close'] <= row['lower_keltner'] + 1.4 * row['atr']):  # Relaxed Keltner
                cash_after_fee = cash * (1 - trading_fee)
                position = cash_after_fee / row['close']
                buy_price = row['close']
                trades.append({
                    'time': row.name,
                    'action': 'buy',
                    'price': row['close'],
                    'position': position,
                    'cash_after': 0,
                    'fee': cash * trading_fee
                })
                cash = 0
                equity[i] = position * row['close']

        # Relaxed cooldown
        if in_cooldown:
            if row['rsi'] > 40:  # Looser than 50
                in_cooldown = False

    # Final value
    final_value = cash + (position * df['close'].iloc[-1] * (1 - trading_fee) if position > 0 else 0)
    if position > 0:
        equity[-1] = final_value

    return trades, final_value, df, equity

def plot_charts(tradingData, trades, df):
    dates = [datetime.fromtimestamp(ts / 1000) for ts in tradingData[0]]
    prices = tradingData[4]
    
    buy_times = [t['time'] for t in trades if t['action'] == 'buy']
    buy_prices = [t['price'] for t in trades if t['action'] == 'buy']
    sell_times = [t['time'] for t in trades if t['action'] in ['sell', 'sell_stop', 'sell_tp', 'sell_final']]
    sell_prices = [t['price'] for t in trades if t['action'] in ['sell', 'sell_stop', 'sell_tp', 'sell_final']]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
    
    ax1.plot(df.index, df['close'], label='Close Price', color='blue')
    ax1.plot(df.index, df['upper_bb'], label='Upper BB', color='orange', linestyle='--')
    ax1.plot(df.index, df['lower_bb'], label='Lower BB', color='orange', linestyle='--')
    ax1.plot(df.index, df['sma'], label='SMA', color='gray', linestyle='-.')
    ax1.plot(df.index, df['upper_keltner'], label='Upper Keltner', color='purple', linestyle=':')
    ax1.plot(df.index, df['lower_keltner'], label='Lower Keltner', color='purple', linestyle=':')
    ax1.plot(df.index, df['middle_keltner'], label='Middle Keltner', color='purple', linestyle='-.')
    ax1.scatter(buy_times, buy_prices, color='green', label='Buys', s=100, marker='^')
    ax1.scatter(sell_times, sell_prices, color='red', label='Sells', s=100, marker='v')
    ax1.set_title('AAVE-USDT Price with Bollinger Bands, Keltner Channels, and Buy/Sell Signals')
    ax1.set_ylabel('Price (USDT)')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(df.index, df['rsi'], label='RSI', color='purple')
    ax2.axhline(70, color='red', linestyle='--', label='Overbought (70)')
    ax2.axhline(30, color='green', linestyle='--', label='Oversold (30)')
    ax2.set_title('RSI')
    ax2.set_ylabel('RSI')
    ax2.legend()
    ax2.grid(True)
    
    ax3.plot(df.index, df['stoch_k'], label='%K', color='blue')
    ax3.plot(df.index, df['stoch_d'], label='%D', color='red')
    ax3.axhline(70, color='red', linestyle='--', label='Overbought (70)')
    ax3.axhline(30, color='green', linestyle='--', label='Oversold (30)')
    ax3.set_title('Stochastic Oscillator')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Stochastic')
    ax3.legend()
    ax3.grid(True)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    equity = []
    cash = 10000
    position = 0
    trade_idx = 0
    for i, (ts, close) in enumerate(zip(tradingData[0], tradingData[4])):
        if trade_idx < len(trades):
            trade_time_ms = int(trades[trade_idx]['time'].timestamp() * 1000)
            if abs(ts - trade_time_ms) < 900000:
                t = trades[trade_idx]
                if t['action'] == 'buy':
                    position = t['position']
                    cash = t['cash_after']
                elif t['action'] in ['sell', 'sell_stop', 'sell_tp', 'sell_final']:
                    cash = t['cash_after']
                    position = 0
                equity.append(cash + (position * close))
                trade_idx += 1
            else:
                equity.append(cash + (position * close))
        else:
            equity.append(cash + (position * close))

    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(dates, equity, label='Equity', color='purple')
    plt.title('Equity Curve')
    plt.xlabel('Time')
    plt.ylabel('Equity (USDT)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.show()

def main():
    file_path = r'../dataHistory/TrainingData/publicData_AAVE-USDT_history_2025-04-01_16_28_02.json'
    tradingData = load_json_data(file_path)
    
    trades, final_value, df, equity = backtest_strategy(tradingData,
                                                       bb_low_threshold=0.0,
                                                       bb_high_threshold=1.0,
                                                       stop_loss_pct=-0.01,
                                                       take_profit_pct=0.02,
                                                       trading_fee=0.001,
                                                       stoch_k_period=14)
    
    plot_charts(tradingData, trades, df)
    
    print(f"\nFinal Value (with hypothetical exit fee if holding): {final_value:.2f} USDT")
    print("\nTrades:")
    for t in trades:
        profit_str = f"Profit %: {t['profit_pct']*100:.2f}%" if 'profit_pct' in t else "N/A"
        print(f"Time: {t['time']}, Action: {t['action']}, Price: {t['price']:.2f}, "
              f"Position: {t['position']:.4f}, Cash After: {t['cash_after']:.2f}, "
              f"{profit_str}, Fee: {t['fee']:.4f}")

if __name__ == "__main__":
    main()