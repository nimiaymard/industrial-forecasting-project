import React from 'react';
import './styles/App.css';
import ForecastChart from './components/ForecastChart';

function App() {
  return (
    <div className="App">
      <h1>Prévision avec LSTM</h1>
      <ForecastChart />
    </div>
  );
}

export default App;
