import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product # step 2
def backtest(
        df,
        buy_threshold=0.03,
        sell_threshold=0.03,
        buy_fraction=0.3,
        sell_fraction=0.5,
        initial_cash=10_000_000,
        fee=0.0005
):
    cash = initial_cash
    shares = 0.0

    wealth_history =[]
    position_history = []

    prices = df['종가'].values
    deviations = df['deviation'].values

    for i in range(len(df)):

        price = prices[i]

        # USe the previous day's signal
        if i > 0:
            signal = deviations[i -1]

            # Buy
            if signal <= -buy_threshold:

                amount = cash * buy_fraction

                shares += amount / (price * (1 + fee))
                cash -= amount

            # Sell    
            elif signal >= sell_threshold:

                quantity = shares * sell_fraction

                cash += quantity * price * (1 - fee)
                shares -= quantity
        wealth = cash + shares * price

        wealth_history.append(wealth)
        position_history.append(shares * price / wealth)
    
    return pd.DataFrame({
        'wealth': wealth_history,
        'position': position_history
    }, index=df.index)

def calculate_metrics(wealth):

    returns = wealth.pct_change().dropna()

    total_return = wealth.iloc[-1] / wealth.iloc[0] - 1

    years = (wealth.index[-1] - wealth.index[0]).days / 365.25

    cagr = (1 + total_return) ** (1 / years) - 1

    drawdown = wealth / wealth.cummax() - 1
    mdd = drawdown.min()

    volatility = returns.std()

    sharpe = (
        np.sqrt(252) * returns.mean() / volatility
        if volatility > 0 else np.nan
    )

    return total_return, cagr, mdd, sharpe

def optimize_strategy(df): # 2

    buy_thresholds = np.arange(1, 11) / 100
    sell_thresholds = np.arange(1, 11) / 100

    buy_fractions = np.arange(1, 11) / 10
    sell_fractions = np.arange(1, 11) / 10

    records = []

    combinations = product(
        buy_thresholds,
        sell_thresholds,
        buy_fractions,
        sell_fractions
    )

    for i, (a, b , fb, fs) in enumerate(combinations):

        result = backtest(
            df,
            buy_threshold=a,
            sell_threshold=b,
            buy_fraction=fb,
            sell_fraction=fs
        )

        total_return, cagr, mdd, sharpe = \
            calculate_metrics(result['wealth'])

        records.append({
            'buy_threshold': a,
            'sell_threshold': b,
            'buy_fraction': fb,
            'sell_fraction': fs,
            'total_return': total_return,
            'cagr': cagr,
            'mdd': mdd,
            'sharpe': sharpe
        })

        if (i + 1) % 1000 == 0:
            print(f'Progress: {i+1}/10000')

    result = pd.DataFrame(records)

    return result.sort_values(
        'cagr',
        ascending=False
    ).reset_index(drop=True)

def main():

    df = pd.read_csv(
        'data/kodex200.csv',
        index_col=0,
        parse_dates=True
    ).sort_index()

    df['ma20'] = df['종가'].rolling(20).mean()

    df['deviation'] = (df['종가'] / df['ma20'] - 1)

    df = df.dropna()

    # 2 Train - test split
    split = int(len(df) * 0.7)

    train = df.iloc[:split]
    test = df.iloc[split:]

    print('\n=== Training Period ===')
    print(train.index[0], '~', train.index[-1])

    print('\n=== Test Period ===')
    print(test.index[0], '~', test.index[-1])

    # Grid search
    optimization = optimize_strategy(train)

    print('\n=== Top 10 Strategies ===')
    print(optimization.head(10).to_string(index=False))

    optimization.to_csv(
        'optimization_results.csv', index=False
    )

    best = optimization.iloc[0]

    print('\n=== OPtimal Parameters ===')
    print(f'Buy threshold: {best['buy_threshold']:.2%}')
    print(f'Sell threshold: {best['sell_threshold']:.2%}')
    print(f'Buy fraction: {best['buy_fraction']:.2%}')
    print(f'Sell fraction: {best['sell_fraction']:.2%}')

    # Out-of-sample backtest
    test_result = backtest(
        test,
        buy_threshold=best['buy_threshold'],
        sell_threshold=best['sell_threshold'],
        buy_fraction=best['buy_fraction'],
        sell_fraction=best['sell_fraction']
    )

    metrics = calculate_metrics(test_result['wealth'])

    print('\n=== Optimized Strategy: Test ===')
    print(f'Total return: {metrics[0]:.2%}')
    print(f'CAGR: {metrics[1]:.2%}')
    print(f'MDD: {metrics[2]:.2%}')
    print(f'Shape ratio: {metrics[3]:.3f}')


    test_benchmarck = (10_000_000 * test['종가'] / test['종가'].iloc[0])

    metrics = calculate_metrics(test_benchmarck)
    
    print('\n=== Buy and Hold ===')
    print(f'Total return: {metrics[0]:.2%}')
    print(f'CAGR: {metrics[1]:.2%}')
    print(f'MDD: {metrics[2]:.2%}')
    print(f'Shape ratio: {metrics[3]:.3f}')
    
    plt.figure(figsize=(12,5))
    plt.plot(result['wealth'], label='Mean Reversion')
    plt.plot(benchmark, label='Buy and Hold')
    plt.ylabel('Portfolio Value (KRW)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    result = backtest(df)

    benchmark = (
        10_000_000 * df['종가'] / df['종가'].iloc[0]
    )

    metrics = calculate_metrics(result['wealth'])
    benchmark_metrics = calculate_metrics(benchmark)

    print('=== Mean Reversion Strategy ===')
    print(f'Total return: {metrics[0]:.2%}')
    print(f'CAGR: {metrics[1]:.2%}')
    print(f'MDD: {metrics[2]:.2%}')
    print(f'Shape ratio: {metrics[3]:.3f}')

    print('\n=== Buy and Hold ===')
    print(f'Total return: {benchmark_metrics[0]:.2%}')
    print(f'CAGR: {benchmark_metrics[1]:.2%}')
    print(f'MDD: {benchmark_metrics[2]:.2%}')
    print(f'Shape ratio: {benchmark_metrics[3]:.3f}')

    plt.figure(figsize=(12,5))
    plt.plot(result['wealth'], label='Mean Reversion')
    plt.plot(benchmark, label='Buy and Hold')
    plt.ylabel('Portfolio Value (KRW)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12,4))
    plt.plot(result['position'] * 100)
    plt.ylabel('Stock Allocation (%)')
    plt.ylim(0, 105)
    plt.grid()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()