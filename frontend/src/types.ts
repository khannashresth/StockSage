export interface StockData {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  sma20: number;
  sma50: number;
  ema20: number;
  macd: number;
  macd_signal: number;
  macd_hist: number;
  rsi: number;
  bb_upper: number;
  bb_lower: number;
  bb_middle: number;
}

export interface CompanyInfo {
  name: string;
  sector: string;
  industry: string;
  description: string;
  website: string;
  marketCap: number;
  peRatio: number;
  eps: number;
  dividendYield: number;
  beta: number;
  fiftyTwoWeekHigh: number;
  fiftyTwoWeekLow: number;
}

export interface StockStats {
  volatility: number;
  avgVolume: number;
  priceChange: number;
  currentRSI: number;
  currentMACD: number;
}

export interface StockResponse {
  success: boolean;
  data: StockData[];
  info: CompanyInfo;
  stats: StockStats;
}

export interface TrendingStock {
  symbol: string;
  name: string;
  change: number;
} 