# Copyright 2024-2025 Jiajun Chen and Jiawei Meng. All Rights Reserved.
#
# This project is developed as part of the coursework for EECS E6895: Advanced Big Data and AI
# at Columbia University.
#
# Project Team:
# - Jiajun Chen
# - Jiawei Meng
#
# Course: EECS E6895: Advanced Big Data and AI
# University: Columbia University
# Project Direction: Fund Manager / M&A Specialist
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================== #
import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta


class StockData:
    def __init__(self, tickers, start_date, validation_date):
        self.tickers = tickers
        self.start_date = start_date
        self.validation_date = validation_date
        self.min_max_scalers = {ticker: MinMaxScaler(feature_range=(0, 1)) for ticker in tickers}

    def download_stock_data(self):
        end_date = datetime.today()
        for ticker in self.tickers:
            print(f'Downloading data for {ticker}')
            df = yf.download(ticker, start=self.start_date, end=end_date)
            data = df[['Close']].copy()
            data.dropna(inplace=True)
            # 修改这里，增加data/目录
            data.to_csv(os.path.join("data", f"{ticker}_data.csv"))
            data.reset_index(inplace=True)
            training_data = data[data['Date'] < self.validation_date].set_index('Date')
            test_data = data[data['Date'] >= self.validation_date].set_index('Date')
            self.min_max_scalers[ticker].fit(training_data)
            setattr(self, f"{ticker}_train", training_data)
            setattr(self, f"{ticker}_test_data", test_data)

    def transform_to_numpy(self, ticker, time_steps):
        training_data = pd.read_csv(os.path.join("data", f"{ticker}_data.csv"), index_col='Date')
        scaled_data = self.min_max_scalers[ticker].fit_transform(training_data)

        x_train, y_train = [], []
        for i in range(time_steps, scaled_data.shape[0]):
            x_train_seq = scaled_data[i - time_steps:i]
            x_train.append(x_train_seq)
            y_train.append(scaled_data[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], time_steps, 1))

        return x_train, y_train









