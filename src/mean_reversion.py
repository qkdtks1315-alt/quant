import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def main():

    df = pd.read_csv(
        'data/kodex200.csv',
        index_col=0,
        parse_dates=True
    ).sort_index()

    df['ma20'] = df['종가'].rolling(20).mean()

    df['deviation'] = (df['종가'] / df['ma20'] - 1)

    df = df.dropna()

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
