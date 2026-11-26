import pandas as pd
import matplotlib.pyplot as plt

def main() -> None:
    df = pd.read_csv('data/kodex200.csv',
                     index_col=0,
                     parse_dates=True)
    
    print('1st 5 row')
    print(df.head())

    print('\nlast 5 row')
    print(df.tail())

    print('\nbasic statistics')
    print(df.describe())

    df['20일 이동평균'] = df['종가'].rolling(20).mean()
    df['60일 이동평균'] = df['종가'].rolling(60).mean()

    df[['종가', '20일 이동평균', '60일 이동평균']].plot(figsize=(12, 6), title='KODEX 200 Price and Moving Averages')

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()