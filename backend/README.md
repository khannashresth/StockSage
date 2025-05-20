# StockSage Backend

The backend service for StockSage, providing real-time stock data, technical analysis, and market insights.

## Features

- Real-time stock data fetching using yfinance
- Technical analysis indicators (SMA, EMA, MACD, RSI, Bollinger Bands)
- Company information and statistics
- Market trending stocks
- Stock search functionality

## Project Structure

```
backend/
├── models/           # Machine learning and forecasting models
├── services/         # Core services (caching, sentiment analysis, etc.)
├── utils/           # Utility functions and helpers
├── data/            # Data storage and management
├── app.py           # Main Flask application
└── requirements.txt # Python dependencies
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the development server:
```bash
python app.py
```

The server will start at http://localhost:5000

## API Endpoints

- `GET /api/stock/<symbol>` - Get stock data and company information
- `GET /api/search?q=<query>` - Search for stocks
- `GET /api/market/trending` - Get trending stocks

## Development

- The project uses Flask for the web framework
- yfinance for stock data
- Technical analysis using the `ta` library
- Data processing with pandas and numpy

## Testing

Run tests using pytest:
```bash
pytest
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request 