# StockSage 📈

A modern, intelligent stock market analysis dashboard that combines real-time data with powerful visualization tools.

![StockSage Dashboard](https://i.imgur.com/placeholder.png)

## Features ✨

- 📊 Real-time stock price tracking
- 📈 Interactive price charts with multiple timeframes
- 🔍 Smart stock search with company information
- 💡 Technical analysis indicators
- 📱 Responsive design for all devices
- 🌙 Dark mode support
- 🚀 Fast and efficient data loading

## Tech Stack 🛠

### Frontend
- React with TypeScript
- Material-UI for beautiful components
- Recharts for interactive charts
- Axios for API communication
- Vite for lightning-fast development

### Backend
- Flask (Python)
- yfinance for real-time stock data
- Pandas for data manipulation
- Flask-CORS for API security

## Getting Started 🚀

### Prerequisites
- Python 3.10+
- Node.js 16+
- npm or yarn

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stocksage.git
cd stocksage
```

2. Set up the backend:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

3. Set up the frontend:
```bash
cd frontend
npm install
npm run dev
```

4. Open your browser and visit `http://localhost:5173`

## API Endpoints 📡

- `GET /api/stock/<symbol>` - Get stock data and company information
- `GET /api/search?q=<query>` - Search for stocks

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.

## License 📝

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments 🙏

- Data provided by Yahoo Finance
- Icons by Material Icons
- Charts powered by Recharts