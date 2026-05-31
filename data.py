import pandas as pd
import pandas_datareader as pdr
from scalecast.Forecaster import Forecaster

def load_housing_data(start='1959-01-01', end='2022-12-31'):
    df = pdr.get_data_fred('HOUST', start=start, end=end)
    f = Forecaster(
        y=df.iloc[:, 0],
        current_dates=df.index,
        future_dates=24,
        test_length=24,
        validation_length=24,
    )
    return f

if __name__ == '__main__':
    f = load_housing_data()
    print(f.y.tail())
    print(f'Series length: {len(f.y)}')