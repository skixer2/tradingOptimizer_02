import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

# Grouped parameters
params = {
    'rsi_period': 6,
    'rsi_type1_threshold': 80,
    'rsi_type2_threshold': 10,
    'macd_fast': 12,
    'macd_slow': 26,
    'sar_initial_af': 0.02,
    'sar_step_af': 0.02,
    'sar_end_af': 0.2,
    'fees': 0.001,
    'stop_loss_ratio': 0.99,
    'take_profit_ratio_type2': 1.01,
    'drop_threshold': 0.980,
    'ema_span': 120,
    'min_bars_buffer': 30,
    'initial_capital': 1000.0
}

def load_data():
    # file_path = '../tradingOptimizer/dataHistory/TrainingData/publicData_AAVE-USDC_history_2025-04-01_08_18_48.json'
    file_path = '../tradingOptimizer/dataHistory/TrainingData/publicData_AAVE-USDT_history_2025-07-23_15_32_12.json' #Bull
    # file_path='../tradingOptimizer/dataHistory/TrainingData/publicData_AAVE-USDT_history_2025-08-03_13_18_11.json' #Bear
    with open(file_path, 'r') as f:
        data_json = json.load(f)
    data = data_json['data']
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'quote_volume2', 'count'])
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})
    return df

def parabolic_sar(df, initial_af, step_af, end_af):
    df = df.copy()
    original_index = df.index.copy()
    df = df.reset_index(drop=True)
    df = df.rename(columns={'high': 'High', 'low': 'Low', 'close': 'Close'})

    df['trend'] = 0
    df['sar'] = 0.0
    df['real sar'] = 0.0
    df['ep'] = 0.0
    df['af'] = 0.0

    if len(df) < 2:
        df['psar'] = np.nan
        df.set_index(original_index, inplace=True)
        df = df.rename(columns={'High': 'high', 'Low': 'low', 'Close': 'close'})
        return df

    # initial values
    df.at[0, 'trend'] = 1  # arbitrary initial trend
    df.at[1, 'trend'] = 1 if df['Close'][1] > df['Close'][0] else -1
    df.at[1, 'sar'] = df['High'][0] if df['trend'][1] > 0 else df['Low'][0]
    df.at[1, 'real sar'] = df['sar'][1]
    df.at[1, 'ep'] = df['High'][1] if df['trend'][1] > 0 else df['Low'][1]
    df.at[1, 'af'] = initial_af

    for i in range(2, len(df)):
        temp = df['sar'][i-1] + df['af'][i-1] * (df['ep'][i-1] - df['sar'][i-1])
        if df['trend'][i-1] < 0:
            df.at[i, 'sar'] = max(temp, df['High'][i-1], df['High'][i-2])
            temp_trend = 1 if df['sar'][i] < df['High'][i] else df['trend'][i-1] - 1
        else:
            df.at[i, 'sar'] = min(temp, df['Low'][i-1], df['Low'][i-2])
            temp_trend = -1 if df['sar'][i] > df['Low'][i] else df['trend'][i-1] + 1

        df.at[i, 'trend'] = temp_trend

        if df['trend'][i] > 0:
            df.at[i, 'ep'] = max(df['High'][i], df['ep'][i-1]) if df['trend'][i-1] > 0 else df['High'][i]
            df.at[i, 'af'] = min(df['af'][i-1] + step_af, end_af) if df['ep'][i] > df['ep'][i-1] else df['af'][i-1]
        else:
            df.at[i, 'ep'] = min(df['Low'][i], df['ep'][i-1]) if df['trend'][i-1] < 0 else df['Low'][i]
            df.at[i, 'af'] = min(df['af'][i-1] + step_af, end_af) if df['ep'][i] < df['ep'][i-1] else df['af'][i-1]

        if abs(df['trend'][i]) == 1:
            df.at[i, 'sar'] = df['ep'][i-1]

        df.at[i, 'real sar'] = df['sar'][i]

    df['psar'] = df['real sar']
    df.set_index(original_index, inplace=True)
    df = df.rename(columns={'High': 'high', 'Low': 'low', 'Close': 'close'})
    return df

def rsi(series, period):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val

def macd(series, fast, slow):
    fast_ema = series.ewm(span=fast, min_periods=1).mean()
    slow_ema = series.ewm(span=slow, min_periods=1).mean()
    macd_line = fast_ema - slow_ema
    return macd_line

def simulate_live_strategy(params):
    df = load_data()
    df.set_index('timestamp', inplace=True)
    df = df.sort_index()

    # Clean low prices
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col].apply(lambda x: np.nan if x < 1 else x).ffill()

    # We'll process row by row, but build df15_history incrementally
    df15_history = pd.DataFrame()  # To build 1-min history

    capital = params['initial_capital']
    holdings = 0.0
    equity = []
    holdings_list = []
    capital_list = []
    equity_pct = []
    price_pct = []
    buy_price = 0.0
    buy_time = None
    buy_type = 0
    last_max = 0.0
    previous_period = None
    prev_psar_above = True
    previous_macd = 0.0

    buys = []
    sells = []
    trades = []

    rsi_buys = []
    sar_buys = []
    macd_sells = []
    sar_sells = []

    trade_history = []  # For stats

    initial_price = None

    # Incremental processing: loop through each 1-min row
    for i in range(len(df)):
        current_row = df.iloc[[i]]  # Current candle as DF

        # Append to history for resampling
        df15_history = pd.concat([df15_history, current_row])

        # Resample up to current time
        df15 = df15_history.resample('15min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
        df15.dropna(inplace=True)

        if len(df15) == 0:
            continue

        df15['rsi6'] = rsi(df15['close'], params['rsi_period'])
        df15['macd_line'] = macd(df15['close'], params['macd_fast'], params['macd_slow'])
        df15 = parabolic_sar(df15, params['sar_initial_af'], params['sar_step_af'], params['sar_end_af'])

        # Ensure columns exist before dropna
        subset = ['rsi6', 'macd_line']
        if 'psar' in df15.columns:
            subset.append('psar')
        df15.dropna(subset=subset, inplace=True)

        if len(df15) == 0:
            continue

        df15['psar_above'] = df15['psar'] > df15['close']

        df15['ema120'] = df15['close'].ewm(span=params['ema_span'], min_periods=1).mean()  # Incremental EMA
        df15['ema120_slope'] = df15['ema120'] - df15['ema120'].shift(1)

        if len(df15) < params['ema_span']:
            continue  # Wait for warm-up

        if initial_price is None:
            initial_price = df15['close'].iloc[-1]

        current_time = current_row.index[0]
        current_price = current_row['close'].iloc[0]
        current_high = current_row['high'].iloc[0]

        if current_price > 0:
            last_good_price = current_price
        else:
            current_price = last_good_price

        current_equity = capital + holdings * current_price
        equity.append(current_equity)
        holdings_list.append(holdings)
        capital_list.append(capital)
        equity_pct.append((current_equity / params['initial_capital']) * 100 if params['initial_capital'] > 0 else 0)
        price_pct.append((current_price / initial_price) * 100 if initial_price > 0 else 0)

        current_period = current_time.floor('15min')

        if previous_period is None or current_period != previous_period:
            last_max = current_high
        else:
            last_max = max(last_max, current_high)
        previous_period = current_period

        just_bought = False

        current_rsi = df15['rsi6'].iloc[-1]
        psar = df15['psar'].iloc[-1]
        macd_current = df15['macd_line'].iloc[-1]
        sar_above = psar > current_price
        switch_to_bull = prev_psar_above and not sar_above
        prev_psar_above = sar_above
        ema120_slope = df15['ema120_slope'].iloc[-1]

        if holdings == 0 and current_price > 0:
            if switch_to_bull and (ema120_slope > 0 or current_rsi < params['rsi_type1_threshold']):
                holdings = (capital / current_price) * (1 - params['fees'])
                capital = 0.0
                buy_price = current_price
                buy_time = current_time
                buy_type = 1
                buys.append((current_time, current_price))
                sar_buys.append((current_time, psar))
                trades.append(f"Buy type 1 at {current_time} price {current_price} equity after {current_equity}")
                just_bought = True
                previous_macd = macd_current
                last_max = current_price
            elif current_rsi < params['rsi_type2_threshold'] and ema120_slope >= 0:
                holdings = (capital / current_price) * (1 - params['fees'])
                capital = 0.0
                buy_price = current_price
                buy_time = current_time
                buy_type = 2
                buys.append((current_time, current_price))
                rsi_buys.append((current_time, current_rsi))
                trades.append(f"Buy type 2 at {current_time} price {current_price} equity after {current_equity}")
                just_bought = True
                previous_macd = macd_current
                last_max = current_price

        if holdings > 0 and current_price > 0 and not just_bought:
            sell = False
            sell_reason = ""
            if buy_type == 1:
                if sar_above:
                    sell = True
                    sell_reason = "SAR above price"
                if macd_current > 0 and macd_current < previous_macd:
                    sell = True
                    sell_reason = "MACD decreasing after positive" if sell_reason == "" else sell_reason + "; MACD decreasing after positive"
                if current_price < buy_price * params['stop_loss_ratio']:
                    sell = True
                    sell_reason = "Price < 99% of buy" if sell_reason == "" else sell_reason + "; Price < 99% of buy"
            elif buy_type == 2:
                if current_price >= buy_price * params['take_profit_ratio_type2']:
                    sell = True
                    sell_reason = "Price >= 101% of buy"
                if current_price < buy_price * params['stop_loss_ratio']:
                    sell = True
                    sell_reason = "Price < 99% of buy" if sell_reason == "" else sell_reason + "; Price < 99% of buy"

            if current_price < last_max * params['drop_threshold']:
                sell = True
                sell_reason = "Price < 99.0% of period max" if sell_reason == "" else sell_reason + "; Price < 99.0% of period max"

            if sell:
                capital = (holdings * current_price) * (1 - params['fees'])
                gain_pct = ((current_price - buy_price) / buy_price) * 100
                trade_history.append((buy_time, buy_price, current_time, current_price, gain_pct))
                holdings = 0.0
                buy_type = 0
                sells.append((current_time, current_price))
                trades.append(f"Sell at {current_time} price {current_price} reason: {sell_reason} equity after {current_equity}")
                if "SAR" in sell_reason:
                    sar_sells.append((current_time, psar))
                if "MACD" in sell_reason:
                    macd_sells.append((current_time, macd_current))

        previous_macd = macd_current

        if holdings > 0 and current_price > 0 and not just_bought:
            sell = False
            sell_reason = ""
            if current_price < buy_price * params['stop_loss_ratio']:
                sell = True
                sell_reason = "1-min stop loss < 99% buy"
            elif buy_type == 2 and current_price >= buy_price * params['take_profit_ratio_type2']:
                sell = True
                sell_reason = "1-min take profit >= 101% buy"
            if current_price < last_max * params['drop_threshold']:
                sell = True
                sell_reason = "1-min < 99.0% period max" if sell_reason == "" else sell_reason + "; 1-min < 99.0% period max"

            if sell:
                capital = (holdings * current_price) * (1 - params['fees'])
                gain_pct = ((current_price - buy_price) / buy_price) * 100
                trade_history.append((buy_time, buy_price, current_time, current_price, gain_pct))
                holdings = 0.0
                buy_type = 0
                sells.append((current_time, current_price))
                trades.append(f"Sell (1-min) at {current_time} price {current_price} reason: {sell_reason} equity after {current_equity}")

    # If still holding at end
    if holdings > 0:
        final_price = df['close'].iloc[-1]
        gain_pct = ((final_price - buy_price) / buy_price) * 100
        trade_history.append((buy_time, buy_price, df.index[-1], final_price, gain_pct))

    df['equity'] = equity
    df['holdings'] = holdings_list
    df['capital'] = capital_list
    df['equity_pct'] = equity_pct
    df['price_pct'] = price_pct

    # Print trades for debugging
    print("Trades executed:")
    for trade in trades:
        print(trade)

    final_equity = df['equity'].iloc[-1]
    print(f"Final Strategy Equity: {final_equity}")

    # Calculate holding strategy equity (buy at start, hold to end, with fees)
    holding_holdings = params['initial_capital'] / initial_price * (1 - params['fees'])
    holding_final_equity = holding_holdings * df['close'].iloc[-1] * (1 - params['fees'])
    print(f"Holding Strategy Final Equity: {holding_final_equity}")
    if final_equity > holding_final_equity:
        print("Strategy beats holding!")
    else:
        print("Strategy does not beat holding.")

    # Statistics
    num_trades = len(trade_history)
    print(f"Number of trades (buy/sell pairs): {num_trades}")

    if num_trades > 0:
        gains = [g for _, _, _, _, g in trade_history]
        avg_gain = np.mean(gains)
        print(f"Average percentage price variation per trade: {avg_gain:.2f}%")

        min_equity = np.min(equity)
        max_equity = np.max(equity)
        print(f"Minimum equity: {min_equity:.2f}")
        print(f"Maximum equity: {max_equity:.2f}")

        equity_series = pd.Series(equity)
        equity_returns = equity_series.pct_change().dropna()
        equity_volatility = equity_returns.std() * np.sqrt(252)  # Annualized
        print(f"Equity volatility (annualized): {equity_volatility:.4f}")

        # Max drawdown
        peak = equity_series.cummax()
        drawdown = (equity_series - peak) / peak
        max_drawdown = drawdown.min()
        print(f"Max drawdown: {max_drawdown * 100:.2f}%")

        # Sharpe ratio (assuming risk-free rate 0)
        sharpe_ratio = equity_returns.mean() / equity_returns.std() * np.sqrt(252)
        print(f"Sharpe ratio: {sharpe_ratio:.2f}")

        # Win rate
        win_rate = len([g for g in gains if g > 0]) / num_trades * 100
        print(f"Win rate: {win_rate:.2f}%")

    # Combined figure with subplots
    fig = plt.figure(figsize=(12, 18))

    # Price subplot
    ax_price = fig.add_subplot(411)
    ax_price.plot(df.index, df['close'], label='Price', color='blue')
    ax_price.plot(df15.index, df15['psar'], label='SAR', color='purple', linestyle='--')
    ax_price.plot(df15.index, df15['ema120'], label='EMA120', color='brown', linestyle='-.')
    # Vertical lines for buys/sells
    for bt in [t for t, p in buys]:
        ax_price.axvline(bt, color='green', linestyle='--', alpha=0.5)
    for st in [t for t, p in sells]:
        ax_price.axvline(st, color='red', linestyle='--', alpha=0.5)
    # SAR-triggered on SAR line
    if sar_buys:
        sar_buy_times, sar_buy_psar = zip(*sar_buys)
        ax_price.scatter(sar_buy_times, sar_buy_psar, color='green', marker='^', s=50, label='SAR Buy')
    if sar_sells:
        sar_sell_times, sar_sell_psar = zip(*sar_sells)
        ax_price.scatter(sar_sell_times, sar_sell_psar, color='red', marker='v', s=50, label='SAR Sell')
    ax_price_pct = ax_price.twinx()
    ax_price_pct.plot(df.index, df['equity_pct'], label='Equity %', color='orange')
    ax_price_pct.plot(df.index, df['price_pct'], label='Price %', color='green', linestyle='--')
    ax_price.set_ylabel('Price (USDT)')
    ax_price_pct.set_ylabel('% Change')
    ax_price.legend(loc='upper left')
    ax_price_pct.legend(loc='upper right')
    ax_price.set_title('Price, SAR, EMA120, Buys/Sells, Equity %, Price %')
    ax_price.grid(True)

    # RSI subplot
    ax_rsi = fig.add_subplot(412, sharex=ax_price)
    ax_rsi.plot(df15.index, df15['rsi6'], label='RSI-6', color='cyan')
    ax_rsi.axhline(70, color='red', linestyle='--')
    ax_rsi.axhline(20, color='green', linestyle='--')
    # RSI-triggered buys
    if rsi_buys:
        rsi_buy_times, rsi_buy_values = zip(*rsi_buys)
        ax_rsi.scatter(rsi_buy_times, rsi_buy_values, color='green', marker='^', s=50, label='RSI Buy')
    ax_rsi.set_ylabel('RSI')
    ax_rsi.legend(loc='upper left')
    ax_rsi.set_title('RSI-6')
    ax_rsi.grid(True)

    # MACD subplot
    ax_macd = fig.add_subplot(413, sharex=ax_price)
    ax_macd.plot(df15.index, df15['macd_line'], label='MACD Line', color='magenta')
    ax_macd.axhline(0, color='black', linestyle='--')
    # MACD-triggered sells
    if macd_sells:
        macd_sell_times, macd_sell_values = zip(*macd_sells)
        ax_macd.scatter(macd_sell_times, macd_sell_values, color='red', marker='v', s=50, label='MACD Sell')
    ax_macd.set_ylabel('MACD')
    ax_macd.legend(loc='upper left')
    ax_macd.set_title('MACD')
    ax_macd.grid(True)

    # Equity subplot
    ax_equity = fig.add_subplot(414, sharex=ax_price)
    ax_equity.plot(df.index, df['equity'], label='Equity', color='blue')
    ax_equity_hold = ax_equity.twinx()
    ax_equity_hold.plot(df.index, df['holdings'], label='Holdings', color='orange', linestyle='--')
    ax_equity_hold.plot(df.index, df['capital'], label='Capital', color='green', linestyle=':')
    ax_equity.set_ylabel('Equity (USDT)')
    ax_equity_hold.set_ylabel('Holdings / Capital')
    ax_equity.legend(loc='upper left')
    ax_equity_hold.legend(loc='upper right')
    ax_equity.set_title('Equity, Holdings, Capital')
    ax_equity.grid(True)

    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.show()

simulate_live_strategy(params)