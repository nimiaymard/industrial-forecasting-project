import React, { useEffect, useState } from 'react';
import { Line } from 'react-chartjs-2';
import { fetchForecast } from '../services/api';

function ForecastChart() {
  const [data, setData] = useState(null);

  useEffect(() => {
    fetchForecast().then(setData);
  }, []);

  if (!data) return <p>Chargement des données...</p>;

  const chartData = {
    labels: data.timestamps,
    datasets: [
      {
        label: 'Prévision',
        data: data.y_pred,
        borderColor: 'blue',
        fill: false,
      },
      {
        label: 'Réel',
        data: data.y_true,
        borderColor: 'green',
        fill: false,
      }
    ]
  };

  return <Line data={chartData} />;
}

export default ForecastChart;
