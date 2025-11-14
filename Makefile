install:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

arima:
	python scripts/train_arima.py --config config.yaml

lstm:
	python scripts/train_lstm.py --config config.yaml

anomaly:
	python scripts/detect_anomalies.py --config config.yaml

eval-arima:
	python scripts/evaluate_forecasts.py --config config.yaml --model arima

eval-lstm:
	python scripts/evaluate_forecasts.py --config config.yaml --model lstm
