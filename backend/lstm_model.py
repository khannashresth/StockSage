import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df = df[['Close', 'Volume']]
    return df.dropna()

def create_dataset(data, sequence_length):
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i - sequence_length:i])
        y.append(data[i, 0])  # Predict Close price
    return np.array(X), np.array(y)

def train_lstm(data, forecast_days=30):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    seq_len = 60
    X, y = create_dataset(scaled_data, seq_len)
    X = np.reshape(X, (X.shape[0], X.shape[1], data.shape[1]))

    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    preds = model.predict(X_test)
    preds = scaler.inverse_transform(np.hstack((preds, np.zeros((len(preds), data.shape[1]-1)))))
    actual = scaler.inverse_transform(np.hstack((y_test.reshape(-1,1), np.zeros((len(y_test), data.shape[1]-1)))))

    # Forecast future
    future = forecast_future(model, scaled_data, forecast_days, data.shape[1], scaler)

    # Plot
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(actual[:,0], label="Actual")
    ax.plot(preds[:,0], label="Predicted")
    ax.axvline(x=len(actual), color="gray", linestyle="--", label="Forecast Start")
    ax.plot(range(len(actual), len(actual)+forecast_days), future, label="Future Forecast")
    ax.set_title("LSTM Stock Forecast")
    ax.legend()
    return fig

def forecast_future(model, data, days, num_features, scaler):
    last_seq = data[-60:]
    preds = []
    for _ in range(days):
        input_seq = last_seq.reshape(1, 60, num_features)
        pred = model.predict(input_seq, verbose=0)[0][0]
        preds.append(pred)
        new_entry = np.zeros((num_features,))
        new_entry[0] = pred
        last_seq = np.vstack((last_seq[1:], new_entry))
    preds = scaler.inverse_transform(np.hstack((np.array(preds).reshape(-1,1), np.zeros((days, num_features-1)))))
    return preds[:,0]