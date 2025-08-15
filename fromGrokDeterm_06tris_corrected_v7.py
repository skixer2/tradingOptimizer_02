import numpy as np
import json
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from scipy.stats import linregress

# Set fixed seed for deterministic results
np.random.seed(42)

def load_json_data(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        ohlcv = sorted(data['data'], key=lambda x: int(x[0]))
        timestamps = np.array([int(row[0]) for row in ohlcv])
        closes = np.array([float(row[4]) for row in ohlcv])
        print(f"Loaded data: {len(timestamps)} points")
        return timestamps, closes
    except Exception as e:
        print(f"Error in load_json_data: {e}")
        raise

def calculate_sharpe_ratio(portfolio_values, risk_free_rate=0.0):
    try:
        portfolio_values = np.array(portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        excess_returns = returns - risk_free_rate
        std_dev = np.std(excess_returns)
        if std_dev == 0:
            return 0
        return np.mean(excess_returns) / std_dev
    except Exception as e:
        print(f"Error in calculate_sharpe_ratio: {e}")
        return 0

def calculate_rsi(prices, period=14):
    try:
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-period:]) if len(gains) >= period else np.mean(gains)
        avg_loss = np.mean(losses[-period:]) if len(losses) >= period else np.mean(losses)
        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - 100 / (1 + rs)
    except Exception as e:
        print(f"Error in calculate_rsi: {e}")
        return 50

def objective(params, timestamps, prices, fee_pct, use_sharpe=False):
    try:
        Rbl, Rb, Rsl, Rs = params
        result = simulate_trading(timestamps, prices, Rbl, Rb, Rsl, Rs, fee_pct, carry_state=False, return_points=True)
        final_value = result['final_value']
        portfolio_values = result['portfolio_values']
        if use_sharpe:
            sharpe = calculate_sharpe_ratio(portfolio_values)
            penalty = 0.1 * (max(0, Rb - 5)**2 + max(0, Rs - 5)**2)
            return - (final_value * (1 + sharpe)) + penalty
        return -final_value
    except Exception as e:
        print(f"Error in objective: {e}")
        return float('inf')

def process_price(price, timestamp, capital, holdings, Rbl, Rb, Rsl, Rs, fee_pct, last_local_min, last_local_max, last_sell_price, last_buy_price, previous_price, trend, return_points, rsi=None, no_trade=False):
    try:
        buy_points = []
        sell_points = []
        trade_profit = 0

        # Update local extrema
        if price < previous_price:
            if trend != 'down':
                trend = 'down'
                last_local_max = previous_price
            if price < last_local_min:
                last_local_min = price
        elif price > previous_price:
            if trend != 'up':
                trend = 'up'
                last_local_min = previous_price
            if price > last_local_max:
                last_local_max = price
        previous_price = price

        total_value = capital + holdings * price

        if no_trade:
            return capital, holdings, total_value, buy_points, sell_points, last_local_min, last_local_max, last_sell_price, last_buy_price, previous_price, trend, trade_profit

        # Apply RSI filter
        buy_allowed = sell_allowed = True
        if rsi is not None:
            buy_allowed = rsi < 60
            sell_allowed = rsi > 40

        # Buy logic
        can_buy = capital >= 0.01 * total_value
        if can_buy and buy_allowed:
            local_trigger = ((price - last_local_min) / last_local_min) * 100 >= Rbl
            global_trigger = ((price - last_sell_price) / last_sell_price) * 100 >= Rb
            if local_trigger or global_trigger:
                buy_amount = 0.10 * total_value
                amount_to_buy = min(buy_amount, capital)
                fee = amount_to_buy * fee_pct / 100
                net_buy = amount_to_buy - fee
                tokens_to_buy = net_buy / price
                holdings += tokens_to_buy
                capital -= amount_to_buy
                if return_points:
                    buy_points.append((timestamp, price))
                last_buy_price = price

        # Sell logic
        can_sell = holdings > 0
        if can_sell and sell_allowed:
            local_drop = ((last_local_max - price) / last_local_max) * 100
            global_drop = ((last_buy_price - price) / last_buy_price) * 100 if last_buy_price else 0
            local_trigger = local_drop >= Rsl
            global_trigger = last_buy_price is not None and global_drop >= Rs
            if local_trigger or global_trigger:
                sell_amount = 0.20 * total_value
                tokens_to_sell = min(sell_amount / price, holdings)
                if tokens_to_sell > 0:
                    gross_sell = tokens_to_sell * price
                    fee = gross_sell * fee_pct / 100
                    net_sell = gross_sell - fee
                    trade_profit = net_sell - (tokens_to_sell * last_buy_price if last_buy_price else 0)
                    holdings -= tokens_to_sell
                    capital += net_sell
                    if return_points:
                        sell_points.append((timestamp, price))
                    last_sell_price = price

        total_value = capital + holdings * price
        return capital, holdings, total_value, buy_points, sell_points, last_local_min, last_local_max, last_sell_price, last_buy_price, previous_price, trend, trade_profit
    except Exception as e:
        print(f"Error in process_price: {e}")
        return capital, holdings, capital + holdings * price, [], [], last_local_min, last_local_max, last_sell_price, last_buy_price, previous_price, trend, 0

def simulate_trading(timestamps, prices, Rbl, Rb, Rsl, Rs, fee_pct, initial_capital=1000, initial_holdings=0.0, initial_state=None, carry_state=False, return_points=True):
    try:
        capital = initial_capital
        holdings = initial_holdings
        portfolio_values = []
        buy_points = [] if return_points else None
        sell_points = [] if return_points else None
        trade_profits = []

        last_local_min = initial_state['last_local_min'] if initial_state else prices[0]
        last_local_max = initial_state['last_local_max'] if initial_state else prices[0]
        last_sell_price = initial_state['last_sell_price'] if initial_state else prices[0]
        last_buy_price = initial_state['last_buy_price'] if initial_state else None
        previous_price = initial_state['previous_price'] if initial_state else prices[0]
        trend = initial_state['trend'] if initial_state else 'unknown'

        for i in range(1, len(prices)):
            price = prices[i]
            timestamp = timestamps[i]
            rsi = calculate_rsi(prices[max(0, i-14):i+1]) if i >= 14 else None
            capital, holdings, total_value, buys, sells, last_local_min, last_local_max, last_sell_price, last_buy_price, previous_price, trend, trade_profit = process_price(
                price, timestamp, capital, holdings, Rbl, Rb, Rsl, Rs, fee_pct, last_local_min, last_local_max, last_sell_price, last_buy_price, previous_price, trend, return_points, rsi
            )
            portfolio_values.append(total_value)
            if return_points:
                buy_points.extend(buys)
                sell_points.extend(sells)
            if trade_profit != 0:
                trade_profits.append(trade_profit)

        final_value = capital + holdings * prices[-1]
        result = {
            'final_value': final_value,
            'portfolio_values': portfolio_values,
            'capital': capital,
            'holdings': holdings,
            'trade_profits': trade_profits
        }
        if carry_state:
            result['state'] = {
                'last_local_min': last_local_min,
                'last_local_max': last_local_max,
                'last_sell_price': last_sell_price,
                'last_buy_price': last_buy_price,
                'previous_price': previous_price,
                'trend': trend
            }
        if return_points:
            result['buy_points'] = buy_points
            result['sell_points'] = sell_points
        return result
    except Exception as e:
        print(f"Error in simulate_trading: {e}")
        return {'final_value': initial_capital, 'portfolio_values': [initial_capital], 'capital': initial_capital, 'holdings': initial_holdings, 'trade_profits': [], 'buy_points': [], 'sell_points': []}

def detect_macro_trend(prices):
    try:
        log_prices = np.log(prices + 1e-8)
        slope, _, _, _, _ = linregress(np.arange(len(log_prices)), log_prices)
        volatility = np.std(np.diff(log_prices))
        if slope > 0.001:
            return 'bull', volatility
        elif slope < -0.001:
            return 'bear', volatility
        else:
            return 'sideways', volatility
    except Exception as e:
        print(f"Error in detect_macro_trend: {e}")
        return 'sideways', 0

def optimize_parameters(timestamps, prices, fee_pct=0.02, previous_params=None, is_initial=False):
    try:
        np.random.seed(42)
        print(f"Starting optimization, is_initial={is_initial}")
        trend, volatility = detect_macro_trend(prices)
        if trend == 'bull':
            bounds = np.array([(0.1, 5.0), (0.2, 5.0), (0.05, 1.0), (0.05, 5.0)])
        elif trend == 'bear':
            bounds = np.array([(0.2, 10.0), (0.2, 5.0), (0.1, 5.0), (0.1, 5.0)])
        else:
            bounds = np.array([(0.1, 4.0), (0.2, 5.0), (0.1, 3.0), (0.1, 5.0)])

        if volatility > 0.05:
            bounds[:, 0] *= 1.2
            bounds[:, 1] *= 0.8

        popsize = 20 if is_initial else 15
        maxiter = 1000 if is_initial else 500  # Revert to maxiter=1000 for initial optimization

        init = None
        if previous_params is not None:
            perturbation = np.random.normal(0, 0.05 * (bounds[:, 1] - bounds[:, 0]))
            init_start = np.clip(previous_params + perturbation, bounds[:, 0], bounds[:, 1])
            random_pop = np.random.uniform(bounds[:, 0], bounds[:, 1], (popsize - 1, 4))
            init = np.vstack([init_start, random_pop])

        def callback(xk, convergence):
            print(f"Optimization iteration, convergence={convergence:.4f}")

        result = differential_evolution(
            objective,
            bounds.tolist(),
            args=(timestamps, prices, fee_pct, False),
            strategy='best1bin',
            popsize=popsize,
            tol=0.01,
            mutation=(0.5, 1),
            recombination=0.7,
            maxiter=maxiter,
            workers=1,
            init=init if init is not None else 'latinhypercube',
            callback=callback,
            disp=True,
            seed=17
        )
        print(f"Optimization completed: Rbl={result.x[0]:.2f}, Rb={result.x[1]:.2f}, Rsl={result.x[2]:.2f}, Rs={result.x[3]:.2f}")
        return result.x
    except Exception as e:
        print(f"Error in optimize_parameters: {e}")
        return [0.1, 0.2, 0.05, 0.05]

def walk_forward_simulation(timestamps, prices, window_size=10080, optimize_every=1440, fee_pct=0.02, initial_capital=1000):
    try:
        print("Starting walk_forward_simulation")
        if len(prices) < window_size:
            raise ValueError("Data too short for window size")

        # Initialize live plot
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.set_title('Portfolio Value')
        ax1.grid(True)
        ax1_right = ax1.twinx()
        ax1_right.set_ylabel('Capital/Holdings ($)')
        ax2.set_ylabel('Price ($)')
        ax2.set_xlabel('Timestamp')
        ax2.grid(True)
        ax2_pct = ax2.twinx()
        ax2_pct.set_ylabel('Percentage Change (%)')

        line_portfolio, = ax1.plot([], [], label='Portfolio Value', color='blue')
        line_capital, = ax1_right.plot([], [], label='Capital', color='green', linestyle='--')
        line_holdings, = ax1_right.plot([], [], label='Holdings ($)', color='purple', linestyle=':')
        line_price, = ax2.plot([], [], label='Price', color='orange')
        line_price_pct, = ax2_pct.plot([], [], label='Price % Change', color='red', linestyle='--')
        line_equity_pct, = ax2_pct.plot([], [], label='Equity % Change', color='blue', linestyle='-.')
        base_price_scatter = ax2.scatter([], [], color='black', label='Base Price', marker='s', s=100)
        buy_scatter = ax2.scatter([], [], color='green', label='Buy', marker='o')
        sell_scatter = ax2.scatter([], [], color='red', label='Sell', marker='x')
        ax1.legend(loc='upper left')
        ax1_right.legend(loc='upper right')
        ax2.legend(loc='upper left')
        ax2_pct.legend(loc='upper right')

        # Initial plot update
        line_portfolio.set_data(timestamps[:1], [initial_capital])
        line_capital.set_data(timestamps[:1], [initial_capital])
        line_holdings.set_data(timestamps[:1], [0])
        line_price.set_data(timestamps[:1], prices[:1])
        ax1.relim()
        ax1.autoscale_view()
        ax1_right.relim()
        ax1_right.autoscale_view()
        ax2.relim()
        ax2.autoscale_view()
        ax2_pct.relim()
        ax2_pct.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()
        print("Initial plot setup complete")

        capital = initial_capital
        holdings = 0.0
        portfolio_values = [initial_capital]
        capital_values = [initial_capital]
        holdings_values = [0.0]
        buy_points = []
        sell_points = []
        trade_profits = []

        # Initialize trading state
        last_local_min = prices[0]
        last_local_max = prices[0]
        last_sell_price = prices[0]
        last_buy_price = None
        previous_price = prices[0]
        trend = 'unknown'

        # Initial optimization
        initial_past_ts = timestamps[:window_size]
        initial_past_prices = prices[:window_size]
        current_params = optimize_parameters(initial_past_ts, initial_past_prices, fee_pct, previous_params=None, is_initial=True)
        print(f"Initial optimization on first {window_size} points: Rbl={current_params[0]:.2f}, Rb={current_params[1]:.2f}, Rsl={current_params[2]:.2f}, Rs={current_params[3]:.2f}")
        previous_params = current_params

        # Update plot after initial optimization
        line_portfolio.set_data(timestamps[:1], [initial_capital])
        line_capital.set_data(timestamps[:1], [initial_capital])
        line_holdings.set_data(timestamps[:1], [0])
        line_price.set_data(timestamps[:1], prices[:1])
        ax1.relim()
        ax1.autoscale_view()
        ax1_right.relim()
        ax1_right.autoscale_view()
        ax2.relim()
        ax2.autoscale_view()
        ax2_pct.relim()
        ax2_pct.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()
        print("Plot updated after initial optimization")

        for i in range(1, len(prices)):
            price = prices[i]
            rsi = calculate_rsi(prices[max(0, i-14):i+1]) if i >= 14 else None
            no_trade = i < window_size
            capital, holdings, total_value, buys, sells, last_local_min, last_local_max, last_sell_price, last_buy_price, previous_price, trend, trade_profit = process_price(
                price, timestamps[i], capital, holdings, *current_params, fee_pct, last_local_min, last_local_max, last_sell_price, last_buy_price, previous_price, trend, True, rsi, no_trade
            )

            portfolio_values.append(total_value)
            capital_values.append(capital)
            holdings_values.append(holdings * price)
            buy_points.extend(buys)
            sell_points.extend(sells)
            if trade_profit != 0:
                trade_profits.append(trade_profit)

            # Re-optimize if needed
            if i >= window_size + optimize_every and (i - window_size) % optimize_every == 0:
                past_ts = timestamps[i - window_size:i]
                past_prices = prices[i - window_size:i]
                current_params = optimize_parameters(past_ts, past_prices, fee_pct, previous_params=previous_params, is_initial=False)
                current_params = 0.95 * previous_params + 0.05 * current_params
                previous_params = current_params
                print(f"Re-optimized at step {i} (using points {i - window_size} to {i-1}): Rbl={current_params[0]:.2f}, Rb={current_params[1]:.2f}, Rsl={current_params[2]:.2f}, Rs={current_params[3]:.2f}")

            # Update live plot every 1000 points
            if i % 1000 == 0 or i == len(prices) - 1:
                try:
                    line_portfolio.set_data(timestamps[:i+1], portfolio_values)
                    line_capital.set_data(timestamps[:i+1], capital_values)
                    line_holdings.set_data(timestamps[:i+1], holdings_values)
                    line_price.set_data(timestamps[:i+1], prices[:i+1])
                    buy_times, buy_vals = zip(*buy_points) if buy_points else ([], [])
                    sell_times, sell_vals = zip(*sell_points) if sell_points else ([], [])
                    buy_scatter.set_offsets(np.array(list(zip(buy_times, buy_vals))) if buy_times else np.empty((0, 2)))
                    sell_scatter.set_offsets(np.array(list(zip(sell_times, sell_vals))) if sell_times else np.empty((0, 2)))
                    if i >= window_size:
                        base_price = prices[window_size]
                        base_equity = portfolio_values[window_size]
                        price_pct = [(p - base_price) / base_price * 100 for p in prices[window_size:i+1]]
                        equity_pct = [(e - base_equity) / base_equity * 100 for e in portfolio_values[window_size:i+1]]
                        line_price_pct.set_data(timestamps[window_size:i+1], price_pct)
                        line_equity_pct.set_data(timestamps[window_size:i+1], equity_pct)
                        base_price_scatter.set_offsets(np.array([[timestamps[window_size], base_price]]))
                        print(f"Step {i}: Base price marker set at timestamp {timestamps[window_size]}, price {base_price:.2f}")
                    else:
                        line_price_pct.set_data([], [])
                        line_equity_pct.set_data([], [])
                        base_price_scatter.set_offsets(np.empty((0, 2)))
                    ax1.relim()
                    ax1.autoscale_view()
                    ax1_right.relim()
                    ax1_right.autoscale_view()
                    ax2.relim()
                    ax2.autoscale_view()
                    ax2_pct.relim()
                    ax2_pct.autoscale_view()
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    print(f"Plot updated at step {i}")
                except Exception as e:
                    print(f"Error in plot update at step {i}: {e}")

            if i % 1000 == 0:
                print(f"Processed step {i}/{len(prices)}")

        final_value = capital + holdings * prices[-1]
        hold_value = initial_capital * prices[-1] / prices[window_size] if prices[window_size] != 0 else initial_capital
        plt.ioff()
        plt.show()
        print(f"Final: Total Buy points={len(buy_points)}, Total Sell points={len(sell_points)}")
        if trade_profits:
            print(f"Average trade profit/loss: {np.mean(trade_profits):.2f}, Total trades: {len(trade_profits)}")
        print(f"Buy-and-Hold Portfolio Value: {hold_value:.2f}")
        print(f"Final Portfolio Value (check): {portfolio_values[-1]:.2f}")
        return final_value, portfolio_values, buy_points, sell_points
    except Exception as e:
        print(f"Error in walk_forward_simulation: {e}")
        plt.ioff()
        plt.close()
        return initial_capital, [initial_capital], [], []

# Example usage
try:
    # file_path = '../tradingOptimizer/dataHistory/TrainingData/publicData_AAVE-USDC_history_2025-04-01_08_18_48.json'
    file_path = '../tradingOptimizer/dataHistory/TrainingData/publicData_AAVE-USDT_history_2025-07-23_15_32_12.json'  #Bull
    # file_path='../tradingOptimizer/dataHistory/TrainingData/publicData_AAVE-USDT_history_2025-08-03_13_18_11.json'    #Bear
    timestamps, prices = load_json_data(file_path)

    window_size = 10080  # ~7 days for 1-minute data
    optimize_every = 1440  # ~1 day
    fee_pct = 0.1

    final_value, portfolio_values, buy_points, sell_points = walk_forward_simulation(
        timestamps, prices, window_size=window_size, optimize_every=optimize_every, fee_pct=fee_pct
    )

    print(f"📈 Final Portfolio Value: {final_value:.2f}")
    print(f"📊 Sharpe Ratio: {calculate_sharpe_ratio(portfolio_values):.4f}")
except Exception as e:
    print(f"Error in main execution: {e}")