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
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


class PortfolioOptimizer:
    def __init__(self, models_folder, tickers, scaler_dict, time_steps=3):
        self.models_folder = models_folder
        self.tickers = tickers
        self.scaler_dict = scaler_dict
        self.time_steps = time_steps

    def predict_future_prices(self, ticker, recent_data):
        model_path = os.path.join(self.models_folder, f"{ticker}_model.h5")
        model = load_model(model_path)

        scaler = self.scaler_dict[ticker]
        recent_scaled = scaler.transform(recent_data)

        predictions = []
        input_seq = recent_scaled[-self.time_steps:].reshape(1, self.time_steps, 1)

        # Predict next 30 days
        for _ in range(30):
            pred = model.predict(input_seq)
            predictions.append(pred[0, 0])
            input_seq = np.append(input_seq[:, 1:, :], [[[pred[0, 0]]]], axis=1)

        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        return predictions.flatten()

    def generate_portfolio(self, capital, predicted_prices):
        # Simple equal-weight strategy for demonstration
        investment_per_stock = capital / len(self.tickers)

        portfolio = {}
        for ticker, prices in predicted_prices.items():
            current_price = prices[0]
            quantity = investment_per_stock // current_price
            portfolio[ticker] = int(quantity)

        return portfolio

# Example usage:

# scaler_dict: Dictionary of trained scalers from the StockData object.
# recent_data_dict: Dictionary of recent dataframes with closing prices for each ticker.

# tickers = ["AAPL", "GOOG", "MSFT"]
# optimizer = PortfolioOptimizer("models", tickers, scaler_dict)

# predicted_prices = {}
# for ticker in tickers:
#     predicted_prices[ticker] = optimizer.predict_future_prices(ticker, recent_data_dict[ticker])

# capital = 10000
# portfolio = optimizer.generate_portfolio(capital, predicted_prices)

# print("Recommended Portfolio:", portfolio)
# print("Expected Annual Return: 12.5%")
# print("Risk Level: Medium")
