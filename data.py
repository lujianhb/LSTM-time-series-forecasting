import os
import pandas as pd
import requests


def get_data():
    file_name = 'btc_usdt.csv'
    quote = 'USDT'
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
    else:
        url = "https://web-api.coinmarketcap.com/v1/cryptocurrency/ohlcv/historical"
        # param = {"convert": "USDT", "slug": "bitcoin", "time_end": "1601510400", "time_start": "1367107200"}
        param = {"convert": quote, "slug": "bitcoin", "time_end": "1677422213", "time_start": "1367107200"}
        content = requests.get(url=url, params=param).json()
        df = pd.json_normalize(content['data']['quotes'])
        df.to_csv(file_name, index=False)
    # Extracting and renaming the important variables
    # df['Date'] = pd.to_datetime(df[f'quote.{quote}.timestamp']).dt.tz_localize(None)
    df['Date'] = pd.to_datetime(df[f'quote.{quote}.timestamp'])
    df['Low'] = df[f'quote.{quote}.low']
    df['High'] = df[f'quote.{quote}.high']
    df['Open'] = df[f'quote.{quote}.open']
    df['Close'] = df[f'quote.{quote}.close']
    df['Volume'] = df[f'quote.{quote}.volume']

    # Drop original and redundant columns
    df = df.drop(
        columns=['time_open', 'time_close', 'time_high', 'time_low', f'quote.{quote}.low', f'quote.{quote}.high',
                 f'quote.{quote}.open', f'quote.{quote}.close', f'quote.{quote}.volume', f'quote.{quote}.market_cap',
                 f'quote.{quote}.timestamp'])
    # Creating a new feature for better representing day-wise values
    # df['Mean'] = (df['Low'] + df['High']) / 2
    # Cleaning the data for any NaN or Null fields
    df = df.dropna()
    # date time typecast
    df['Date'] = pd.to_datetime(df['Date'])
    df.index = df['Date']
    # normalizing the exogeneous variables
    return df
