import axios from 'axios';

const BASE_URL = 'http://localhost:5000/api';

export interface StockData {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface MarketDepth {
  bids: [number, number][];  // [price, quantity][]
  asks: [number, number][];
}

export const api = {
  getStockData: async (symbol: string, timeframe: string = '1d'): Promise<StockData[]> => {
    const response = await axios.get(`${BASE_URL}/stock/${symbol}?timeframe=${timeframe}`);
    return response.data;
  },

  getMarketDepth: async (symbol: string): Promise<MarketDepth> => {
    const response = await axios.get(`${BASE_URL}/market-depth/${symbol}`);
    return response.data;
  },

  searchStocks: async (query: string): Promise<string[]> => {
    const response = await axios.get(`${BASE_URL}/search?q=${query}`);
    return response.data;
  }
}; 