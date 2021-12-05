import time
from datetime import date, datetime, timedelta
import requests
import pandas as pd

API_BASE = 'https://api.binance.com/api/v3/'

LABELS = ['open_time','open','high','low','close','volume','close_time','quote_asset_volume','number_of_trades','taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore']

def get_batch(symbol, interval, start_time, limit=1000):
    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': start_time,
        'limit': limit
    }
    try:
        response = requests.get(f'{API_BASE}klines', params)
    except requests.exceptions.ConnectionError:
        print('Waiting for 5 mins...')
        time.sleep(5 * 60)
        return get_batch(symbol, interval, start_time, limit)
    if response.status_code == 200:
        return pd.DataFrame(response.json(), columns=LABELS)
    print(f'Got erroneous response back: {response}')
    return pd.DataFrame([])


def candle_data_to_csv(base, quote, interval):
    try:
        batches = [pd.read_csv(f'{base}-{quote}-{interval}.csv')]
        last_timestamp = batches[-1]['open_time'].max()
    except Exception as e:
        batches = [pd.DataFrame([], columns=LABELS)]
        last_timestamp = 0
    previous_timestamp = None

    while previous_timestamp != last_timestamp:
        if date.fromtimestamp(last_timestamp / 1000) >= date.today():
            break
        previous_timestamp = last_timestamp
        new_batch = get_batch(
            symbol=base+quote,
            interval=interval,
            start_time=last_timestamp+1
        )
        if new_batch.empty:
            break
        last_timestamp = new_batch['open_time'].max()
        if previous_timestamp == last_timestamp:
            break
        batches.append(new_batch)
        last_datetime = datetime.fromtimestamp(last_timestamp / 1000)
        covering_spaces = 20 * ' '
        print(datetime.now(), base, quote, interval, str(last_datetime)+covering_spaces, end='\r', flush=True)
    df = pd.concat(batches, ignore_index=True)
    if len(batches) > 1:
        df.to_csv(f'{base}-{quote}-{interval}.csv', index=False)
        return True
    return 0


def main():
    #intervals = ['1m','5m', '15m', '30m', '1h',  '4h', '1d', '1w']
    intervals = ['1h','1d', '1w']
    for interval in intervals:
        new_lines = candle_data_to_csv(base='BTC', quote='USDT',interval=interval)


if __name__ == '__main__':
    main()