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
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

class LongShortTermMemory:
    def __init__(self, project_folder):
        self.project_folder = project_folder

    def get_defined_metrics(self):
        return [tf.keras.metrics.MeanSquaredError(name='MSE')]

    def get_callback(self):
        callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=3, mode='min', verbose=1
        )
        return callback

    def create_model(self, x_train):
        model = Sequential([
            LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1], 1)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True),
            Dropout(0.5),
            LSTM(units=50),
            Dropout(0.5),
            Dense(units=1)
        ])
        model.summary()
        return model
