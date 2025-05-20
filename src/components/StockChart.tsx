import React from 'react';
import { Paper, Typography, CircularProgress, Box } from '@mui/material';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';
import { useStockData } from '../hooks/useStock';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface StockChartProps {
  symbol?: string;
}

const StockChart: React.FC<StockChartProps> = ({ symbol = 'AAPL' }) => {
  const { data: stockData, isLoading, error } = useStockData(symbol);

  if (isLoading) {
    return (
      <Paper sx={{ p: 2, height: 400, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
        <CircularProgress />
      </Paper>
    );
  }

  if (error || !stockData) {
    return (
      <Paper sx={{ p: 2, height: 400, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
        <Typography color="error">Error loading stock data</Typography>
      </Paper>
    );
  }

  const chartData = {
    labels: stockData.map(d => new Date(d.timestamp).toLocaleDateString()),
    datasets: [
      {
        label: `${symbol} Price`,
        data: stockData.map(d => d.close),
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.1
      }
    ]
  };

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      title: {
        display: true,
        text: `${symbol} Stock Price`,
      },
    },
    scales: {
      y: {
        beginAtZero: false,
      }
    },
    maintainAspectRatio: false
  };

  return (
    <Paper sx={{ p: 2, height: 400 }}>
      <Box sx={{ height: '100%' }}>
        <Line data={chartData} options={options} />
      </Box>
    </Paper>
  );
};

export default StockChart; 