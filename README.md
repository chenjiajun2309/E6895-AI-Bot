# Fund Manager / M&A Specialist AI Project

## EECS E6895: Advanced Big Data and AI
### Columbia University | Midterm Project

**Project Team:**
- **Jiajun Chen** jc6397@columbia.edu
- **Jiawei Meng**

---

### Overview
This project is developed as part of the coursework for EECS E6895: Advanced Big Data and AI at Columbia University. Our objective is to create an AI-driven Fund Manager and M&A Specialist capable of analyzing market data, predicting stock prices using advanced machine learning techniques, and optimizing investment portfolios.

---

### Project Features
- **Market Data Analysis:**
  - Retrieves historical and real-time data for stocks (e.g., AAPL, GOOG, MSFT, ^FTSE).
  - Utilizes yfinance and other financial data APIs.

- **Predictive Modeling:**
  - Implements LSTM (Long Short-Term Memory) models for accurate stock price forecasting.
  - Performs model training and inference on multiple stock tickers simultaneously.

- **Portfolio Optimization:**
  - Provides automated portfolio recommendations based on predicted prices and available capital.
  - Integrates MinMaxScaler for data normalization and preprocessing.

- **Interactive QA Chatbot:**
  - Employs Hugging Face's AutoModelForCausalLM to create a Retrieval-Augmented Generation (RAG) chatbot.
  - Offers natural language responses and investment guidance.

---

### Technology Stack
- Python
- TensorFlow & Keras
- Transformers (Hugging Face)
- FAISS (Vector Database)
- Sentence Transformers (Embedding)
- Pandas & NumPy
- scikit-learn (MinMaxScaler)

---

### Project Structure
```
.
├── data (Market data CSV files)
├── models (Trained LSTM models)
├── stock_prediction_lstm.py
├── stock_prediction_numpy.py
├── portfolio_optimizer.py
└── chatbot.py
```

---

### Usage
- Ensure all dependencies are installed (see `requirements.txt`).
- Run scripts sequentially to perform data preprocessing, model training, portfolio optimization, and chatbot interaction.

---

### License
This project is licensed under the Apache License, Version 2.0. See [LICENSE](http://www.apache.org/licenses/LICENSE-2.0) for details.

---

© 2024-2025 Jiajun Chen and Jiawei Meng. All Rights Reserved.
