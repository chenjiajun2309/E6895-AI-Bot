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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


class MultiStockLSTM:
    def __init__(self, project_folder, time_steps=3):
        self.project_folder = project_folder
        self.time_steps = time_steps

    def create_lstm_model(self, input_shape):
        model = Sequential([
            Input(shape=input_shape),
            LSTM(100, return_sequences=True),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
        return model

    def train_multiple_stocks(self, stock_data_obj, tickers, epochs=50, batch_size=10):
        trained_models = {}

        for ticker in tickers:
            print(f"Training model for {ticker}")

            # Prepare data for each ticker
            x_train, y_train = stock_data_obj.transform_to_numpy(ticker, self.time_steps)

            # Create and train model
            model = self.create_lstm_model(input_shape=(x_train.shape[1], 1))

            early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)

            model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[early_stop])

            # Save the model
            model_filename = os.path.join(self.project_folder, f"{ticker}_model.h5")
            model.save(model_filename)
            trained_models[ticker] = model_filename

            print(f"Model for {ticker} saved at {model_filename}")

        return trained_models
