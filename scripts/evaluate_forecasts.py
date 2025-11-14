import argparse, pandas as pd, os, joblib, torch
from industrial_forecasting.utils.config import load_config
from industrial_forecasting.data import load_series, train_test_split_series
from industrial_forecasting.evaluate import mae, rmse
from industrial_forecasting.models.arima import ARIMAForecaster
from industrial_forecasting.features import create_supervised_from_series
from industrial_forecasting.models.lstm import LSTMRegressor, predict_lstm
import numpy as np

def eval_arima(cfg, s):
    train, test = train_test_split_series(s, cfg['data']['train_ratio'])
    model = ARIMAForecaster.load('models/arima_model.pkl')
    yhat = model.forecast(len(test))
    return test.values, yhat.values

def eval_lstm(cfg, s):
  
    train, test = train_test_split_series(s, cfg['data']['train_ratio'])
    window = int(cfg['lstm']['window_size'])
    X_all, y_all = create_supervised_from_series(train.append(test), window)
    y_test = y_all[-len(test):]
    X_test = X_all[-len(test):]
    model = LSTMRegressor(1, cfg['lstm']['hidden_size'], cfg['lstm']['num_layers'])
    model.load_state_dict(torch.load('models/lstm_model.pt', map_location='cpu'))
    model.eval()
    yhat = predict_lstm(model, X_test)
    return y_test, yhat

def main(cfg_path, model_name):
    cfg = load_config(cfg_path)
    s = load_series(cfg['data']['raw_path'], cfg['data']['datetime_col'], cfg['data']['value_col'], cfg['data']['freq'])

    if model_name == 'arima':
        y_true, y_pred = eval_arima(cfg, s)
    elif model_name == 'lstm':
        y_true, y_pred = eval_lstm(cfg, s)
    else:
        raise ValueError("--model doit être 'arima' ou 'lstm'")

    print(f"MAE: {mae(y_true, y_pred):.3f}")
    print(f"RMSE: {rmse(y_true, y_pred):.3f}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--model', required=True, choices=['arima','lstm'])
    args = ap.parse_args()
    main(args.config, args.model)
