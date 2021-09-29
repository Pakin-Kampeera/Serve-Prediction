# Stress Prediction

## Prediction

- Word2Vec + TF-IDF (Feature Extraction Model)
- Logistic Regression (Classification Model)

## Technology

- FastAPI (Restful framework)

## Prerequisite

- Python3

## Enable Virtual Environment

#### Windows user

```
python3 -m venv serve_pred
serve_pred\Scripts\activate.bat
pip install -r requirements.txt
```

#### Mac user

```
python3 -m venv serve_pred
source serve_pred/bin/activate
pip install -r requirements.txt
```

## Start

```
python server.py
```

## Test API

http://127.0.0.1:8000/docs
