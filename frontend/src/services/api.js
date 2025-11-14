export async function fetchForecast() {
  const response = await fetch('/data/forecast_lstm.csv');
  const text = await response.text();
  const rows = text.trim().split('\n').slice(1); // Skip header
  const timestamps = [];
  const y_true = [];
  const y_pred = [];

  for (const row of rows) {
    const [date, trueVal, predVal] = row.split(',');
    timestamps.push(date);
    y_true.push(parseFloat(trueVal));
    y_pred.push(parseFloat(predVal));
  }

  return { timestamps, y_true, y_pred };
}
