"""
List of NSE (National Stock Exchange of India) stocks
Symbols are suffixed with .NS for Yahoo Finance
"""

import pandas as pd
import requests
import logging
from typing import Dict, List
from datetime import datetime
import os
import json
from cache_service import cached
from config import get_cache_ttl
import time

logger = logging.getLogger(__name__)

class NSEStockManager:
    def __init__(self):
        self.base_url = "https://www.nseindia.com"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }
        self._stocks_cache = {}
        self._last_update = None
        self._retries = 3
        self._retry_delay = 5  # seconds
    
    @cached(ttl=86400)  # Cache for 24 hours
    def get_all_stocks(self) -> Dict[str, str]:
        """Get all NSE stocks with their company names."""
        stocks = {}
        
        # Try multiple data sources in order of preference
        data_sources = [
            self._load_from_json,
            self._load_from_csv,
            self._fetch_from_nse,
            self._fetch_from_alternative_source,
            self._get_fallback_stocks
        ]
        
        for source in data_sources:
            try:
                stocks = source()
                if stocks:
                    logger.info(f"Successfully loaded stocks from {source.__name__}")
                    # Save to both formats for redundancy
                    self._save_to_json(stocks)
                    self._save_to_csv(stocks)
                    return stocks
            except Exception as e:
                logger.warning(f"Failed to load stocks from {source.__name__}: {str(e)}")
                continue
        
        # If all sources fail, return fallback
        return self._get_fallback_stocks()
    
    def _load_from_json(self) -> Dict[str, str]:
        """Load stocks from local JSON file."""
        try:
            json_path = 'nse_stocks.json'
            if os.path.exists(json_path):
                if datetime.fromtimestamp(os.path.getmtime(json_path)) > datetime.now().timestamp() - 86400:
                    with open(json_path, 'r') as f:
                        return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading stocks from JSON: {str(e)}")
            return {}
    
    def _save_to_json(self, stocks: Dict[str, str]):
        """Save stocks to local JSON file."""
        try:
            with open('nse_stocks.json', 'w') as f:
                json.dump(stocks, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving stocks to JSON: {str(e)}")
    
    def _load_from_csv(self) -> Dict[str, str]:
        """Load stocks from local CSV file."""
        try:
            csv_path = 'nse_stocks.csv'
            if os.path.exists(csv_path):
                if datetime.fromtimestamp(os.path.getmtime(csv_path)) > datetime.now().timestamp() - 86400:
                    df = pd.read_csv(csv_path)
                    return dict(zip(df['Symbol'], df['Name']))
            return {}
        except Exception as e:
            logger.error(f"Error loading stocks from CSV: {str(e)}")
            return {}
    
    def _save_to_csv(self, stocks: Dict[str, str]):
        """Save stocks to local CSV file."""
        try:
            df = pd.DataFrame(list(stocks.items()), columns=['Symbol', 'Name'])
            df.to_csv('nse_stocks.csv', index=False)
        except Exception as e:
            logger.error(f"Error saving stocks to CSV: {str(e)}")
    
    def _fetch_from_nse(self) -> Dict[str, str]:
        """Fetch stocks from NSE website with retries."""
        for attempt in range(self._retries):
            try:
                url = f"{self.base_url}/market-data/list-of-securities"
                response = requests.get(url, headers=self.headers, timeout=10)
                
                if response.status_code == 200:
                    df = pd.read_html(response.text)[0]
                    stocks = {}
                    
                    for _, row in df.iterrows():
                        symbol = str(row['Symbol']).strip()
                        name = str(row['Company Name']).strip()
                        if symbol and name:  # Only add valid entries
                            stocks[f"{symbol}.NS"] = name
                    
                    if stocks:  # Only return if we got valid data
                        return stocks
                
                logger.warning(f"Attempt {attempt + 1}/{self._retries} failed to fetch NSE data")
                if attempt < self._retries - 1:
                    time.sleep(self._retry_delay)
                    
            except Exception as e:
                logger.error(f"Error fetching from NSE (attempt {attempt + 1}/{self._retries}): {str(e)}")
                if attempt < self._retries - 1:
                    time.sleep(self._retry_delay)
        
        return {}
    
    def _fetch_from_alternative_source(self) -> Dict[str, str]:
        """Fetch stocks from alternative source (Yahoo Finance API)."""
        try:
            # Using Yahoo Finance API as backup
            url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?formatted=true&lang=en-US&region=IN&scrIds=all_stocks_in_nse&count=2000"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                stocks = {}
                
                for quote in data.get('finance', {}).get('result', [{}])[0].get('quotes', []):
                    symbol = quote.get('symbol')
                    name = quote.get('shortName') or quote.get('longName')
                    if symbol and name and symbol.endswith('.NS'):
                        stocks[symbol] = name
                
                return stocks
            
            return {}
            
        except Exception as e:
            logger.error(f"Error fetching from alternative source: {str(e)}")
            return {}
    
    def _get_fallback_stocks(self) -> Dict[str, str]:
        """Return a fallback list of major NSE stocks."""
        return {
            'RELIANCE.NS': 'Reliance Industries Ltd.',
            'TCS.NS': 'Tata Consultancy Services Ltd.',
            'HDFCBANK.NS': 'HDFC Bank Ltd.',
            'INFY.NS': 'Infosys Ltd.',
            'HINDUNILVR.NS': 'Hindustan Unilever Ltd.',
            'ICICIBANK.NS': 'ICICI Bank Ltd.',
            'SBIN.NS': 'State Bank of India',
            'HDFC.NS': 'Housing Development Finance Corporation Ltd.',
            'BHARTIARTL.NS': 'Bharti Airtel Ltd.',
            'ITC.NS': 'ITC Ltd.',
            'KOTAKBANK.NS': 'Kotak Mahindra Bank Ltd.',
            'LT.NS': 'Larsen & Toubro Ltd.',
            'AXISBANK.NS': 'Axis Bank Ltd.',
            'ASIANPAINT.NS': 'Asian Paints Ltd.',
            'MARUTI.NS': 'Maruti Suzuki India Ltd.',
            'WIPRO.NS': 'Wipro Ltd.',
            'HCLTECH.NS': 'HCL Technologies Ltd.',
            'ULTRACEMCO.NS': 'UltraTech Cement Ltd.',
            'TITAN.NS': 'Titan Company Ltd.',
            'BAJFINANCE.NS': 'Bajaj Finance Ltd.',
            'SUNPHARMA.NS': 'Sun Pharmaceutical Industries Ltd.',
            'TATAMOTORS.NS': 'Tata Motors Ltd.',
            'ADANIPORTS.NS': 'Adani Ports and Special Economic Zone Ltd.',
            'POWERGRID.NS': 'Power Grid Corporation of India Ltd.',
            'NTPC.NS': 'NTPC Ltd.',
            'ONGC.NS': 'Oil & Natural Gas Corporation Ltd.',
            'GRASIM.NS': 'Grasim Industries Ltd.',
            'COALINDIA.NS': 'Coal India Ltd.',
            'HINDALCO.NS': 'Hindalco Industries Ltd.',
            'TATASTEEL.NS': 'Tata Steel Ltd.'
        }
    
    def get_indices(self) -> Dict[str, str]:
        """Get major NSE indices."""
        return {
            '^NSEI': 'NIFTY 50',
            '^BSESN': 'SENSEX',
            '^NSEBANK': 'BANK NIFTY',
            '^CNXIT': 'NIFTY IT',
            '^NSEMDCP': 'NIFTY MIDCAP 100',
            '^NSMIDCP': 'NIFTY MIDCAP 50',
            '^CRSLDX': 'NIFTY 500',
            '^NSEAUTO': 'NIFTY AUTO',
            '^NSEFMCG': 'NIFTY FMCG',
            '^NSEPHARM': 'NIFTY PHARMA'
        }
    
    def get_stock_info(self, symbol: str) -> Dict:
        """Get detailed information about a stock."""
        try:
            # Remove .NS suffix for NSE API
            clean_symbol = symbol.replace('.NS', '')
            url = f"{self.base_url}/api/quote-equity?symbol={clean_symbol}"
            
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                data = response.json()
                return {
                    'symbol': symbol,
                    'name': data.get('info', {}).get('companyName'),
                    'sector': data.get('info', {}).get('industry'),
                    'market_cap': data.get('securityInfo', {}).get('marketCap'),
                    'face_value': data.get('securityInfo', {}).get('faceValue'),
                    'pe_ratio': data.get('securityInfo', {}).get('pe'),
                    'eps': data.get('securityInfo', {}).get('eps'),
                    'high52': data.get('securityInfo', {}).get('high52'),
                    'low52': data.get('securityInfo', {}).get('low52')
                }
            return {}
            
        except Exception as e:
            logger.error(f"Error fetching stock info: {str(e)}")
            return {}

# Initialize NSE stock manager
nse_manager = NSEStockManager()

# Get all NSE stocks
NSE_STOCKS = nse_manager.get_all_stocks() 