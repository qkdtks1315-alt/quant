import pandas as pd
import matplotlib.pyplot as plt

def calculate_mdd(cumulative_return: pd.Series) -> float:
    peak = cumulative_return.cummax()
    drawdown = cumulative_return / peak - 1
    return drawdown.min()

def main() -> None:
    df = pd.read_csv(
        'data/kodex200.csv',
        index_col=0,
        parse_dates=True
    )

    # 일간 수익률
    df['daily_return'] = df['종가'].pct_change()
    
    # 이동평균
    df['ma20'] = df['종가'].rolling(20).mean()
    df['ma60'] = df['종가'].rolling(60).mean()

    # 오늘 이동평균으로 오늘 매매하면 미래 정보를 쓰는 문제가 생김 따라서 전날 신호를 다음 날 수익률에 적용
    df['signal'] = (df['ma20'] > df['ma60']).astype(int)
    df['position'] = df['signal'].shift(1)

    # 전략 수익률
    df['strategy_return'] = df['daily_return'] * df['position']

    # 누적 자산
    df['buy_hold'] = (1 + df['daily_return'].fillna(0)).cumprod()
    df['ma_strategy'] = (1 + df['strategy_return'].fillna(0)).cumprod()

    # 결과 계산
    buy_hold_total = df['buy_hold'].iloc[-1] - 1
    strategy_total = df['ma_strategy'].iloc[-1] - 1

    buy_hold_mdd = calculate_mdd(df['buy_hold'])
    strategy_mdd = calculate_mdd(df['ma_strategy'])

    # 포지션 변경 횟수
    trades = df['position'].diff().abs().fillna(0).sum()

    print('=== Backtest Result ===')
    print(f'Buy & Hold total return: {buy_hold_total:.2%}')
    print(f'MA strategy total return: {strategy_total:.2%}')
    print(f'Buy & Hold MDD: {buy_hold_mdd:.2%}')
    print(f'MA strategy MDD: {strategy_mdd:.2%}')
    print(f'Position changes: {int(trades)}')

    initial_capital = 1_000_000

    print(f'\n=== Initial Capital: {initial_capital} KRW ===')
    print(f'Buy & Hold final value: '
          f'{initial_capital * df['buy_hold'].iloc[-1]:,.0f} KRW')
    print(f'MA strategy final value: '
          f'{initial_capital * df['ma_strategy'].iloc[-1]:,.0f} KRW')
    
    # plot
    df[['buy_hold', 'ma_strategy']].plot(figsize=(12, 6), title='KODEX 200: Buy & Hold vs MA Strategy')

    plt.xlabel('Date')
    plt.ylabel('Cumulative Wealth')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()