import { useEffect, useState } from 'react';
import { Box, Typography, CircularProgress, Paper, Grid } from '@mui/material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  ReferenceLine
} from 'recharts';
import axios from 'axios';
import { StockData } from '../types';

interface Props {
  symbol: string;
}

const StockChart = ({ symbol }: Props) => {
  const [data, setData] = useState<StockData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showIndicators, setShowIndicators] = useState({
    sma: true,
    ema: true,
    bb: true,
    macd: false,
    rsi: false
  });

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);
        const response = await axios.get(`http://localhost:5000/api/stock/${symbol}`);
        if (response.data.success) {
          setData(response.data.data);
        } else {
          setError('Failed to fetch stock data');
        }
      } catch (err) {
        setError('Error connecting to server');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [symbol]);

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Typography color="error" align="center">
        {error}
      </Typography>
    );
  }

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        {symbol} Stock Price History
      </Typography>
      
      <Grid container spacing={2} sx={{ mb: 2 }}>
        <Grid item>
          <Paper sx={{ p: 1 }}>
            <Typography variant="body2" color="text.secondary">
              Price: ${data[data.length - 1]?.close.toFixed(2)}
            </Typography>
          </Paper>
        </Grid>
        <Grid item>
          <Paper sx={{ p: 1 }}>
            <Typography variant="body2" color="text.secondary">
              RSI: {data[data.length - 1]?.rsi.toFixed(2)}
            </Typography>
          </Paper>
        </Grid>
        <Grid item>
          <Paper sx={{ p: 1 }}>
            <Typography variant="body2" color="text.secondary">
              MACD: {data[data.length - 1]?.macd.toFixed(2)}
            </Typography>
          </Paper>
        </Grid>
      </Grid>

      <Box sx={{ width: '100%', height: 400 }}>
        <ResponsiveContainer>
          <LineChart
            data={data}
            margin={{
              top: 5,
              right: 30,
              left: 20,
              bottom: 5,
            }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" />
            <YAxis yAxisId="left" />
            <YAxis yAxisId="right" orientation="right" />
            <Tooltip />
            <Legend />
            
            {/* Price Lines */}
            <Line
              yAxisId="left"
              type="monotone"
              dataKey="close"
              stroke="#8884d8"
              dot={false}
              name="Price"
            />
            
            {/* Technical Indicators */}
            {showIndicators.sma && (
              <Line
                yAxisId="left"
                type="monotone"
                dataKey="sma20"
                stroke="#ff7300"
                dot={false}
                name="SMA 20"
              />
            )}
            {showIndicators.ema && (
              <Line
                yAxisId="left"
                type="monotone"
                dataKey="ema20"
                stroke="#00C49F"
                dot={false}
                name="EMA 20"
              />
            )}
            {showIndicators.bb && (
              <>
                <Line
                  yAxisId="left"
                  type="monotone"
                  dataKey="bb_upper"
                  stroke="#888888"
                  dot={false}
                  name="BB Upper"
                />
                <Line
                  yAxisId="left"
                  type="monotone"
                  dataKey="bb_lower"
                  stroke="#888888"
                  dot={false}
                  name="BB Lower"
                />
              </>
            )}
          </LineChart>
        </ResponsiveContainer>
      </Box>

      {/* MACD Chart */}
      {showIndicators.macd && (
        <Box sx={{ width: '100%', height: 200, mt: 2 }}>
          <ResponsiveContainer>
            <LineChart
              data={data}
              margin={{
                top: 5,
                right: 30,
                left: 20,
                bottom: 5,
              }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line
                type="monotone"
                dataKey="macd"
                stroke="#8884d8"
                dot={false}
                name="MACD"
              />
              <Line
                type="monotone"
                dataKey="macd_signal"
                stroke="#ff7300"
                dot={false}
                name="Signal"
              />
              <ReferenceLine y={0} stroke="#666" />
            </LineChart>
          </ResponsiveContainer>
        </Box>
      )}

      {/* RSI Chart */}
      {showIndicators.rsi && (
        <Box sx={{ width: '100%', height: 200, mt: 2 }}>
          <ResponsiveContainer>
            <LineChart
              data={data}
              margin={{
                top: 5,
                right: 30,
                left: 20,
                bottom: 5,
              }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis domain={[0, 100]} />
              <Tooltip />
              <Legend />
              <Line
                type="monotone"
                dataKey="rsi"
                stroke="#8884d8"
                dot={false}
                name="RSI"
              />
              <ReferenceLine y={70} stroke="#ff7300" />
              <ReferenceLine y={30} stroke="#00C49F" />
            </LineChart>
          </ResponsiveContainer>
        </Box>
      )}
    </Box>
  );
};

export default StockChart; 