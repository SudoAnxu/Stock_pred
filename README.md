
# 📈 AAPL Stock Price Prediction using CNN

This project demonstrates how to build a deep learning model using **1D Convolutional Neural Networks (CNN)** to predict Apple's stock closing prices. The model uses historical stock data fetched via the `yfinance` API, processes it into time-series format, and trains a CNN model to predict future prices.

---

## 🚀 Features

* Fetches real-time historical stock data from Yahoo Finance.
* Preprocesses and normalizes data using `MinMaxScaler`.
* Converts time series into supervised learning format.
* Trains a CNN on stock price sequences.
* Evaluates performance using Mean Squared Error (MSE).
* Visualizes predicted vs actual stock prices.

---

## 🛠️ Technologies Used

* Python
* [yfinance](https://pypi.org/project/yfinance/)
* NumPy, Pandas
* Scikit-learn (MinMaxScaler, train\_test\_split)
* TensorFlow / Keras (Sequential API, Conv1D, Dense, Dropout)
* Matplotlib (for result visualization)

---

## 📊 Dataset

Data is downloaded directly from Yahoo Finance using the `yfinance` package for the ticker:

```
AAPL (Apple Inc.)
Date Range: 2015-01-01 to 2023-01-01
```

---

## 🧠 Model Architecture

* Conv1D Layer (64 filters, kernel size 3)
* Dropout Layer (0.2)
* Conv1D Layer (64 filters, kernel size 3)
* Flatten Layer
* Dense Layer (50 units, ReLU)
* Dense Output Layer (1 unit)

Loss function: `Mean Squared Error`
Optimizer: `Adam`

---

## 📈 Training Performance

* Trained for **20 epochs**.
* Achieved a **final validation loss \~ 0.0014**.
* Captured trends effectively in test data.
* Visualized actual vs predicted prices on a time-series graph.

---

## 📌 Instructions

1. **Install dependencies**:

   ```bash
   pip install yfinance numpy pandas scikit-learn tensorflow matplotlib
   ```

2. **Run the script**:

   ```bash
   python stock_price_prediction.py
   ```

3. **Output**:

   * Model training summary.
   * Test set loss.
   * Matplotlib plot comparing predicted vs actual prices.

---

## 📉 Result Visualization

* Blue Line: Actual Stock Prices
* Red Line: Predicted Stock Prices

The plot clearly shows that the CNN model captures the stock price patterns over time and can be used for trend forecasting.

---

## 📌 Notes

* This is a demonstration for educational purposes and does **not** offer financial advice.
* For better generalization, try experimenting with other models like LSTM, GRU, or hybrid models.
* Expand the dataset with more features like volume, high, low, or use multi-step forecasting.

---

## 🙌 Author

**Priyangshu Karmakar**
B.Tech, IIT Kharagpur
📧 [priyangshukarmakar@gmail.com](mailto:priyangshu.0718k@gmail.com)
🔗 [GitHub](https://github.com/SudoAnxu)

