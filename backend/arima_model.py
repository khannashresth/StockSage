import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

def run_arima(data, forecast_days=30):
    series = data['Close']
    model = ARIMA(series, order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=forecast_days)

    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(series, label="Historical")
    ax.plot(range(len(series), len(series)+forecast_days), forecast, label="Forecast")
    ax.set_title("ARIMA Forecast")
    ax.legend()
    return fig