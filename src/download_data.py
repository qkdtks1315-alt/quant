from pathlib import Path
from pykrx import stock

def main() -> None:
    ticker = '069500' # KODEX 200
    start_date = '20200101'
    end_date = '20260720'

    dataframe = stock.get_market_ohlcv_by_date(
        start_date, end_date, ticker
    )

    if dataframe.empty:
        raise RuntimeError('No data')
    
    output_directory = Path('data')
    output_directory.mkdir(exist_ok=True)

    output_path = output_directory / 'kodex200.csv'
    dataframe.to_csv(output_path, encoding='utf-8-sig')

    print(dataframe.head())
    print()
    print(dataframe.tail())
    print(f'\nSaved : {output_path}')

if __name__ == '__main__':
    main()