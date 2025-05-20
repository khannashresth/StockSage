import websocket
import json
import threading
import queue
import logging
from typing import Dict, List, Optional, Callable
from datetime import datetime
import pandas as pd
import yfinance as yf
import time
from random import uniform
from config import API_KEYS
import requests

# Get API keys from config
FINNHUB_API_KEY = API_KEYS['FINNHUB_API_KEY']
ALPHA_VANTAGE_API_KEY = API_KEYS['ALPHA_VANTAGE_API_KEY']

# Optional imports with fallbacks
try:
    from alpha_vantage.timeseries import TimeSeries
    from alpha_vantage.fundamentaldata import FundamentalData
    ALPHA_VANTAGE_AVAILABLE = True
except ImportError:
    ALPHA_VANTAGE_AVAILABLE = False

try:
    from finnhub import Client
    FINNHUB_AVAILABLE = True
except ImportError:
    FINNHUB_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

class LiveMarketData:
    def __init__(self):
        self.ws: Optional[websocket.WebSocketApp] = None
        self.data_queue = queue.Queue()
        self.subscribed_symbols: List[str] = []
        self.callbacks: Dict[str, List[Callable]] = {}
        self.last_prices: Dict[str, float] = {}
        self.connection_active = False
        
        # Rate limiting parameters
        self.last_reconnect_attempt = 0
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.base_delay = 1  # Base delay in seconds
        
        # Initialize various data providers if available
        if ALPHA_VANTAGE_AVAILABLE:
            try:
                self.alpha_vantage_ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY)
                self.alpha_vantage_fd = FundamentalData(key=ALPHA_VANTAGE_API_KEY)
            except:
                self.alpha_vantage_ts = None
                self.alpha_vantage_fd = None
        else:
            self.alpha_vantage_ts = None
            self.alpha_vantage_fd = None
        
        if FINNHUB_AVAILABLE:
            try:
                self.finnhub_client = Client(api_key=FINNHUB_API_KEY)
            except:
                self.finnhub_client = None
        else:
            self.finnhub_client = None
        
        # Redis for caching
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
                self.redis_client.ping()  # Test connection
            except:
                self.redis_client = None
                logging.warning("Redis server not available. Using in-memory caching.")
        else:
            self.redis_client = None
            logging.warning("Redis package not installed. Using in-memory caching.")
        
        # In-memory cache as fallback
        self.memory_cache = {}
        
        # Add market depth cache
        self.depth_cache = {}
        self.last_depth_update = {}
        self.depth_update_interval = 5  # seconds
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
        # Start WebSocket connection
        self.connect()
    
    def setup_logging(self):
        """Configure logging for the data system."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('market_data.log'),
                logging.StreamHandler()
            ]
        )
    
    def connect(self):
        """Establish WebSocket connection with rate limiting."""
        if time.time() - self.last_reconnect_attempt < 1:  # Minimum 1 second between connection attempts
            return
        
        def on_message(ws, message):
            try:
                data = json.loads(message)
                self.data_queue.put(data)
                self.process_message(data)
            except Exception as e:
                self.logger.error(f"Error processing message: {str(e)}")
        
        def on_error(ws, error):
            if "429" in str(error):
                self.logger.warning("Rate limit exceeded. Implementing backoff...")
                self.reconnect_attempts += 1
            else:
                self.logger.error(f"WebSocket error: {str(error)}")
            self.connection_active = False
        
        def on_close(ws, close_status_code, close_msg):
            self.logger.info("WebSocket connection closed")
            self.connection_active = False
            self.reconnect()
        
        def on_open(ws):
            self.logger.info("WebSocket connection established")
            self.connection_active = True
            self.reconnect_attempts = 0  # Reset counter on successful connection
            self.subscribe_symbols()
        
        self.ws = websocket.WebSocketApp(
            f"wss://ws.finnhub.io?token={FINNHUB_API_KEY}",
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        # Start WebSocket connection in a separate thread
        self.ws_thread = threading.Thread(target=self.ws.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()
        
        self.last_reconnect_attempt = time.time()
    
    def reconnect(self):
        """Attempt to reconnect with exponential backoff."""
        if not self.connection_active:
            current_time = time.time()
            time_since_last_attempt = current_time - self.last_reconnect_attempt
            
            # If we've exceeded max attempts, wait longer
            if self.reconnect_attempts >= self.max_reconnect_attempts:
                if time_since_last_attempt < 60:  # Wait at least 1 minute after max attempts
                    self.logger.info("Too many reconnection attempts. Waiting before next attempt...")
                    return
                self.reconnect_attempts = 0  # Reset counter after waiting
            
            # Calculate delay with exponential backoff and jitter
            delay = min(300, self.base_delay * (2 ** self.reconnect_attempts))  # Cap at 5 minutes
            jitter = uniform(-0.1, 0.1) * delay  # Add ±10% jitter
            delay += jitter
            
            if time_since_last_attempt < delay:
                return  # Too soon to retry
            
            self.logger.info(f"Attempting to reconnect (attempt {self.reconnect_attempts + 1})...")
            try:
                self.connect()
                self.last_reconnect_attempt = current_time
                self.reconnect_attempts += 1
            except Exception as e:
                self.logger.error(f"Reconnection failed: {str(e)}")
    
    def subscribe_symbols(self, symbols: Optional[List[str]] = None):
        """Subscribe to market data for specified symbols."""
        if symbols:
            self.subscribed_symbols.extend(symbols)
        
        if self.connection_active and self.subscribed_symbols:
            for symbol in self.subscribed_symbols:
                self.ws.send(json.dumps({'type': 'subscribe', 'symbol': symbol}))
    
    def unsubscribe_symbols(self, symbols: Optional[List[str]] = None):
        """Unsubscribe from market data for specified symbols."""
        if not symbols:
            symbols = self.subscribed_symbols.copy()
        
        for symbol in symbols:
            if symbol in self.subscribed_symbols:
                self.ws.send(json.dumps({'type': 'unsubscribe', 'symbol': symbol}))
                self.subscribed_symbols.remove(symbol)
    
    def process_message(self, data: Dict):
        """Process incoming WebSocket messages."""
        try:
            if data.get('type') == 'trade':
                symbol = data.get('data', [{}])[0].get('s')
                if symbol:
                    price_data = data.get('data', [{}])[0]
                    self.last_prices[symbol] = price_data.get('p', 0)
                    
                    # Update market depth if available
                    if 'v' in price_data:  # Volume information available
                        if symbol not in self.depth_cache:
                            self.depth_cache[symbol] = {'bids': [], 'asks': [], 'timestamp': time.time(), 'source': 'websocket'}
                        
                        # Update the appropriate side based on trade type
                        if price_data.get('side', '').lower() == 'buy':
                            self._update_depth_side(self.depth_cache[symbol]['bids'], price_data['p'], price_data['v'], 'bid')
                        else:
                            self._update_depth_side(self.depth_cache[symbol]['asks'], price_data['p'], price_data['v'], 'ask')
                    
                    # Trigger callbacks
                    self.trigger_callbacks(symbol, price_data)
        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}")
    
    def register_callback(self, symbol: str, callback: Callable):
        """Register a callback function for a symbol."""
        if symbol not in self.callbacks:
            self.callbacks[symbol] = []
        self.callbacks[symbol].append(callback)
    
    def trigger_callbacks(self, symbol: str, data: Dict):
        """Trigger registered callbacks for a symbol."""
        if symbol in self.callbacks:
            for callback in self.callbacks[symbol]:
                try:
                    callback(data)
                except Exception as e:
                    self.logger.error(f"Error in callback: {str(e)}")
    
    def cache_data(self, key: str, data: Dict):
        """Cache data with fallback to in-memory storage."""
        try:
            if self.redis_client:
                self.redis_client.setex(
                    key,
                    3600,  # 1 hour expiration
                    json.dumps(data)
                )
            else:
                self.memory_cache[key] = {
                    'data': data,
                    'timestamp': datetime.now()
                }
        except Exception as e:
            self.logger.error(f"Caching error: {str(e)}")
    
    def get_cached_data(self, key: str) -> Optional[Dict]:
        """Retrieve cached data with fallback to in-memory storage."""
        try:
            if self.redis_client:
                data = self.redis_client.get(key)
                return json.loads(data) if data else None
            else:
                cached = self.memory_cache.get(key)
                if cached:
                    # Check if cache is still valid (less than 1 hour old)
                    if (datetime.now() - cached['timestamp']).seconds < 3600:
                        return cached['data']
                    else:
                        del self.memory_cache[key]
                return None
        except Exception as e:
            self.logger.error(f"Cache retrieval error: {str(e)}")
            return None
    
    def get_real_time_price(self, symbol: str) -> Optional[Dict]:
        """Get real-time price data with fallback mechanisms."""
        try:
            # Try Finnhub first
            if FINNHUB_AVAILABLE and self.finnhub_client:
                try:
                    quote = self.finnhub_client.quote(symbol)
                    if quote and quote.get('c'):  # Current price exists
                        return {
                            'price': quote['c'],
                            'change': quote['d'],
                            'change_percent': quote['dp'],
                            'high': quote['h'],
                            'low': quote['l'],
                            'open': quote['o'],
                            'previous_close': quote['pc'],
                            'timestamp': datetime.now().timestamp(),
                            'source': 'finnhub'
                        }
                except Exception as e:
                    self.logger.warning(f"Finnhub real-time data failed: {str(e)}")

            # Fallback to Yahoo Finance
            try:
                ticker = yf.Ticker(symbol)
                live_data = ticker.history(period='1d', interval='1m')
                if not live_data.empty:
                    latest = live_data.iloc[-1]
                    return {
                        'price': latest['Close'],
                        'change': latest['Close'] - live_data.iloc[0]['Open'],
                        'change_percent': ((latest['Close'] - live_data.iloc[0]['Open']) / live_data.iloc[0]['Open']) * 100,
                        'high': latest['High'],
                        'low': latest['Low'],
                        'open': live_data.iloc[0]['Open'],
                        'previous_close': live_data.iloc[0]['Open'],
                        'timestamp': datetime.now().timestamp(),
                        'source': 'yahoo'
                    }
            except Exception as e:
                self.logger.warning(f"Yahoo Finance fallback failed: {str(e)}")

            # Last resort: Alpha Vantage
            if ALPHA_VANTAGE_AVAILABLE and self.alpha_vantage_ts:
                try:
                    data, _ = self.alpha_vantage_ts.get_quote_endpoint(symbol)
                    if data:
                        return {
                            'price': float(data['05. price']),
                            'change': float(data['09. change']),
                            'change_percent': float(data['10. change percent'].strip('%')),
                            'high': float(data['03. high']),
                            'low': float(data['04. low']),
                            'open': float(data['02. open']),
                            'previous_close': float(data['08. previous close']),
                            'timestamp': datetime.now().timestamp(),
                            'source': 'alpha_vantage'
                        }
                except Exception as e:
                    self.logger.warning(f"Alpha Vantage fallback failed: {str(e)}")

            return None
        except Exception as e:
            self.logger.error(f"Error fetching real-time price: {str(e)}")
            return None
    
    def get_historical_data(self, symbol: str, interval: str = '1d', period: str = '1y') -> pd.DataFrame:
        """Get historical data with fallback sources."""
        try:
            # Try yfinance first
            data = yf.download(symbol, period=period, interval=interval)
            if not data.empty:
                return data
            
            # Fallback to Alpha Vantage
            data, _ = self.alpha_vantage_ts.get_daily(symbol=symbol, outputsize='full')
            return pd.DataFrame(data).transpose()
        except Exception as e:
            self.logger.error(f"Error fetching historical data: {str(e)}")
            return pd.DataFrame()
    
    def get_fundamental_data(self, symbol: str) -> Dict:
        """Get fundamental data with fallback to yfinance."""
        try:
            # Try Alpha Vantage first
            if self.alpha_vantage_fd:
                try:
                    overview = self.alpha_vantage_fd.get_company_overview(symbol)[0]
                    return overview
                except:
                    pass
            
            # Fallback to yfinance
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'Name': info.get('longName', ''),
                'Description': info.get('longBusinessSummary', ''),
                'Sector': info.get('sector', ''),
                'Industry': info.get('industry', ''),
                'Market Cap': info.get('marketCap', 0),
                'PE Ratio': info.get('trailingPE', 0),
                'Forward PE': info.get('forwardPE', 0),
                'PB Ratio': info.get('priceToBook', 0),
                'Dividend Yield': info.get('dividendYield', 0),
                '52 Week High': info.get('fiftyTwoWeekHigh', 0),
                '52 Week Low': info.get('fiftyTwoWeekLow', 0)
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching fundamental data: {str(e)}")
            return {}
    
    def get_market_sentiment(self, symbol: str) -> Dict:
        """Get market sentiment with fallback behavior."""
        try:
            if self.finnhub_client:
                sentiment = self.finnhub_client.news_sentiment(symbol)
                return sentiment
            
            # Fallback to basic sentiment
            return {
                'sentiment': {
                    'score': 0,
                    'label': 'Neutral'
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching market sentiment: {str(e)}")
            return {}
    
    def get_market_depth(self, symbol: str) -> Dict:
        """Get market depth data with fallbacks."""
        try:
            current_time = time.time()
            
            # Check cache first
            if symbol in self.depth_cache:
                last_update = self.last_depth_update.get(symbol, 0)
                if current_time - last_update < self.depth_update_interval:
                    return self.depth_cache[symbol]
            
            depth_data = {
                'bids': [],
                'asks': [],
                'timestamp': current_time,
                'source': None
            }
            
            # Try Finnhub first
            if FINNHUB_AVAILABLE and self.finnhub_client:
                try:
                    book = self.finnhub_client.level2_market_data(symbol)
                    if book and 'bids' in book and 'asks' in book:
                        depth_data.update({
                            'bids': [{'price': b[0], 'volume': b[1]} for b in book['bids']],
                            'asks': [{'price': a[0], 'volume': a[1]} for a in book['asks']],
                            'source': 'finnhub'
                        })
                        self.depth_cache[symbol] = depth_data
                        self.last_depth_update[symbol] = current_time
                        return depth_data
                except Exception as e:
                    self.logger.warning(f"Finnhub market depth failed: {str(e)}")
            
            # Fallback to Yahoo Finance bid/ask
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                if info:
                    depth_data.update({
                        'bids': [{'price': info.get('bid', 0), 'volume': info.get('bidSize', 0)}],
                        'asks': [{'price': info.get('ask', 0), 'volume': info.get('askSize', 0)}],
                        'source': 'yahoo'
                    })
                    self.depth_cache[symbol] = depth_data
                    self.last_depth_update[symbol] = current_time
                    return depth_data
            except Exception as e:
                self.logger.warning(f"Yahoo Finance market depth failed: {str(e)}")
            
            # If all fails, return empty structure
            return depth_data
        except Exception as e:
            self.logger.error(f"Error fetching market depth: {str(e)}")
            return {'bids': [], 'asks': [], 'timestamp': time.time(), 'source': None}
    
    def _update_depth_side(self, side_data: List[Dict], price: float, volume: int, side: str):
        """Update market depth data for a given side."""
        # Find existing price level or insert new one
        for level in side_data:
            if abs(level['price'] - price) < 0.0001:  # Price levels match
                level['volume'] += volume
                return
        
        # Add new price level
        side_data.append({'price': price, 'volume': volume})
        
        # Sort based on side (bids descending, asks ascending)
        side_data.sort(key=lambda x: x['price'], reverse=(side == 'bid'))
        
        # Keep only top 10 levels
        while len(side_data) > 10:
            side_data.pop()
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.should_stop = True
        if self.ws:
            self.ws.close()
        if self.redis_client:
            try:
                self.redis_client.close()
            except:
                pass 