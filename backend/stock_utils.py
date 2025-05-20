import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

def calculate_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators for screening."""
    df = data.copy()
    
    # Price data
    df['Close_Prev'] = df['Close'].shift(1)
    df['High_Prev'] = df['High'].shift(1)
    df['Low_Prev'] = df['Low'].shift(1)
    
    # Basic Performance Metrics
    df['Daily_Return'] = (df['Close'] - df['Close_Prev']) / df['Close_Prev'] * 100
    df['Weekly_Return'] = df['Close'].pct_change(periods=5) * 100
    df['Monthly_Return'] = df['Close'].pct_change(periods=20) * 100
    df['Yearly_Return'] = df['Close'].pct_change(periods=252) * 100
    
    # Volume Analysis
    df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']
    
    # Moving Averages
    for period in [5, 10, 20, 50, 100, 200]:
        df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
        df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
    
    # RSI (14-period)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    
    # Average True Range (ATR)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close_Prev'])
    low_close = np.abs(df['Low'] - df['Close_Prev'])
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    # Volatility
    df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
    
    # Price Channels
    df['Upper_Channel'] = df['High'].rolling(20).max()
    df['Lower_Channel'] = df['Low'].rolling(20).min()
    
    # Gap Analysis
    df['Gap'] = df['Open'] - df['Close_Prev']
    df['Gap_Percentage'] = (df['Gap'] / df['Close_Prev']) * 100
    
    return df

def screen_stocks(df: pd.DataFrame, criteria: Dict) -> bool:
    """Screen stocks based on technical and fundamental criteria."""
    try:
        latest = df.iloc[-1]
        
        for criterion, value in criteria.items():
            if criterion == 'price_above':
                if latest['Close'] <= value:
                    return False
            elif criterion == 'price_below':
                if latest['Close'] >= value:
                    return False
            elif criterion == 'volume_above':
                if latest['Volume'] <= value:
                    return False
            elif criterion == 'rsi_above':
                if latest['RSI'] <= value:
                    return False
            elif criterion == 'rsi_below':
                if latest['RSI'] >= value:
                    return False
            elif criterion == 'sma_cross':
                short_ma = f'SMA_{value[0]}'
                long_ma = f'SMA_{value[1]}'
                if not (df[short_ma].iloc[-2] < df[long_ma].iloc[-2] and 
                       df[short_ma].iloc[-1] > df[long_ma].iloc[-1]):
                    return False
            elif criterion == 'macd_cross':
                if not (df['MACD'].iloc[-2] < df['Signal_Line'].iloc[-2] and 
                       df['MACD'].iloc[-1] > df['Signal_Line'].iloc[-1]):
                    return False
            elif criterion == 'volatility_above':
                if latest['Volatility'] <= value:
                    return False
            elif criterion == 'volatility_below':
                if latest['Volatility'] >= value:
                    return False
            elif criterion == 'return_above':
                if latest[f'{value[0]}_Return'] <= value[1]:
                    return False
            elif criterion == 'volume_surge':
                if latest['Volume_Ratio'] <= value:
                    return False
            elif criterion == 'bb_position':
                if value == 'above' and latest['Close'] <= latest['BB_Upper']:
                    return False
                elif value == 'below' and latest['Close'] >= latest['BB_Lower']:
                    return False
            elif criterion == 'trend':
                if value == 'uptrend':
                    if not (latest['Close'] > latest['SMA_20'] > latest['SMA_50']):
                        return False
                elif value == 'downtrend':
                    if not (latest['Close'] < latest['SMA_20'] < latest['SMA_50']):
                        return False
        return True
    except Exception:
        return False

def get_stock_info(ticker: str) -> Dict:
    """Get detailed stock information."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        return {
            'name': info.get('longName', ''),
            'sector': info.get('sector', ''),
            'industry': info.get('industry', ''),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'pb_ratio': info.get('priceToBook', 0),
            'dividend_yield': info.get('dividendYield', 0),
            'beta': info.get('beta', 0),
            'avg_volume': info.get('averageVolume', 0),
            'description': info.get('longBusinessSummary', '')
        }
    except:
        return {} 