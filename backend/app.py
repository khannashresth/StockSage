from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

app = Flask(__name__)
CORS(app)

def calculate_technical_indicators(df):
    # Calculate SMA
    sma_20 = SMAIndicator(close=df['Close'], window=20)
    sma_50 = SMAIndicator(close=df['Close'], window=50)
    df['SMA20'] = sma_20.sma_indicator()
    df['SMA50'] = sma_50.sma_indicator()

    # Calculate EMA
    ema_20 = EMAIndicator(close=df['Close'], window=20)
    df['EMA20'] = ema_20.ema_indicator()

    # Calculate MACD
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()

    # Calculate RSI
    rsi = RSIIndicator(close=df['Close'])
    df['RSI'] = rsi.rsi()

    # Calculate Bollinger Bands
    bb = BollingerBands(close=df['Close'])
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Lower'] = bb.bollinger_lband()
    df['BB_Middle'] = bb.bollinger_mavg()

    return df

@app.route('/api/stock/<symbol>', methods=['GET'])
def get_stock_data(symbol):
    try:
        # Get parameters
        period = request.args.get('period', '1y')  # default 1 year
        interval = request.args.get('interval', '1d')  # default daily

        # Fetch stock data
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period, interval=interval)
        
        # Calculate technical indicators
        hist = calculate_technical_indicators(hist)
        
        # Process the data
        data = []
        for index, row in hist.iterrows():
            data.append({
                'date': index.strftime('%Y-%m-%d'),
                'open': row['Open'],
                'high': row['High'],
                'low': row['Low'],
                'close': row['Close'],
                'volume': row['Volume'],
                'sma20': row['SMA20'],
                'sma50': row['SMA50'],
                'ema20': row['EMA20'],
                'macd': row['MACD'],
                'macd_signal': row['MACD_Signal'],
                'macd_hist': row['MACD_Hist'],
                'rsi': row['RSI'],
                'bb_upper': row['BB_Upper'],
                'bb_lower': row['BB_Lower'],
                'bb_middle': row['BB_Middle']
            })
        
        # Get company info
        info = stock.info
        company_info = {
            'name': info.get('longName', ''),
            'sector': info.get('sector', ''),
            'industry': info.get('industry', ''),
            'description': info.get('longBusinessSummary', ''),
            'website': info.get('website', ''),
            'marketCap': info.get('marketCap', 0),
            'peRatio': info.get('trailingPE', 0),
            'eps': info.get('trailingEps', 0),
            'dividendYield': info.get('dividendYield', 0),
            'beta': info.get('beta', 0),
            'fiftyTwoWeekHigh': info.get('fiftyTwoWeekHigh', 0),
            'fiftyTwoWeekLow': info.get('fiftyTwoWeekLow', 0)
        }
        
        # Calculate additional statistics
        stats = {
            'volatility': hist['Close'].pct_change().std() * np.sqrt(252),
            'avgVolume': hist['Volume'].mean(),
            'priceChange': ((hist['Close'][-1] - hist['Close'][0]) / hist['Close'][0]) * 100,
            'currentRSI': hist['RSI'].iloc[-1],
            'currentMACD': hist['MACD'].iloc[-1]
        }
        
        return jsonify({
            'success': True,
            'data': data,
            'info': company_info,
            'stats': stats
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/search', methods=['GET'])
def search_stocks():
    query = request.args.get('q', '').upper()
    
    # In a real application, you would have a proper stock database
    # For now, we'll return some sample data
    sample_stocks = [
        {'symbol': 'AAPL', 'name': 'Apple Inc.'},
        {'symbol': 'MSFT', 'name': 'Microsoft Corporation'},
        {'symbol': 'GOOGL', 'name': 'Alphabet Inc.'},
        {'symbol': 'AMZN', 'name': 'Amazon.com Inc.'},
        {'symbol': 'META', 'name': 'Meta Platforms Inc.'},
        {'symbol': 'TSLA', 'name': 'Tesla Inc.'},
        {'symbol': 'NVDA', 'name': 'NVIDIA Corporation'},
        {'symbol': 'JPM', 'name': 'JPMorgan Chase & Co.'},
        {'symbol': 'V', 'name': 'Visa Inc.'},
        {'symbol': 'WMT', 'name': 'Walmart Inc.'}
    ]
    
    # Filter stocks based on query
    results = [
        stock for stock in sample_stocks
        if query in stock['symbol'] or query.lower() in stock['name'].lower()
    ]
    
    return jsonify({
        'success': True,
        'results': results
    })

@app.route('/api/market/trending', methods=['GET'])
def get_trending_stocks():
    # In a real application, this would analyze market data
    # For now, return some sample trending stocks
    trending = [
        {'symbol': 'AAPL', 'name': 'Apple Inc.', 'change': 2.5},
        {'symbol': 'MSFT', 'name': 'Microsoft Corporation', 'change': 1.8},
        {'symbol': 'GOOGL', 'name': 'Alphabet Inc.', 'change': -0.5},
        {'symbol': 'AMZN', 'name': 'Amazon.com Inc.', 'change': 3.2},
        {'symbol': 'TSLA', 'name': 'Tesla Inc.', 'change': -1.2}
    ]
    
    return jsonify({
        'success': True,
        'trending': trending
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000) 