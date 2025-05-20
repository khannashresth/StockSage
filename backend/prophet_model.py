import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

def run_prophet(data, forecast_days=30):
    df = data.reset_index()[['Date', 'Close']]
    df.columns = ['ds', 'y']
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)

    fig = model.plot(forecast)
    return fig