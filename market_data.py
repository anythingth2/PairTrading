from typing import List

from pathlib import Path
import numpy as np
import pandas as pd
import tqdm

class MarketData:
    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)

        self.__create_symbol_hour_mapping_path()
        self.__create_symbol_day_mapping_path()

    def __create_symbol_hour_mapping_path(self):
        price_paths = list(self.dataset_dir.glob('*/price/*.parquet'))
        symbols = list(map(lambda path: path.stem, price_paths))
        self.SYMBOL_HOUR_TO_PATHS = dict(zip(symbols, price_paths))
    
    def __create_symbol_day_mapping_path(self):
        price_paths = list(self.dataset_dir.glob('*/price_daily/*.parquet'))
        symbols = list(map(lambda path: path.stem, price_paths))
        self.SYMBOL_DAY_TO_PATHS = dict(zip(symbols, price_paths))

    def __clean(self, prices: pd.Series) -> pd.Series:
        prices = prices.replace({0: np.nan})
        return prices

    def read(self, symbol: str, timeframe: str = 'day') -> pd.DataFrame:
        if timeframe == 'hour':
            file_path = self.SYMBOL_HOUR_TO_PATHS[symbol]
        elif timeframe == 'day':
            file_path = self.SYMBOL_DAY_TO_PATHS[symbol]
        else:
            raise NotImplementedError
        df = pd.read_parquet(file_path)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df

    def read_batch(self, 
        symbols: List[str], 
        price_source: str = 'close',
        timeframe: str = 'day',
        ignore_not_found: bool = False
    ) -> pd.DataFrame:
        prices = []
        for symbol in tqdm.tqdm(symbols):
            try:
                ohlcv_df = self.read(symbol, timeframe=timeframe)
            except:
                continue
            price_df = ohlcv_df[[price_source]].copy()
            price_df.rename(columns={price_source: symbol}, inplace=True)
            prices.append(price_df)

        price_df = pd.concat(prices, axis=1)
        price_df = price_df.apply(self.__clean, axis=0)
        return price_df