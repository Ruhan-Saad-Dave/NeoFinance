# Transformer Model Architecture

NeoFinance uses two custom-built PyTorch transformer models for stock price prediction:

| File | Purpose | Training data |
|------|---------|---------------|
| `stock_prediction_transformer.py` | Daily price prediction | Yahoo Finance daily OHLCV |
| `stock_prediction_transformer_intraday.py` | Intraday price prediction | Alpha Vantage 1-minute OHLCV |

Both share the same architecture. Only the data source and trained weights differ.

---

## Table of Contents

1. [Why a Transformer?](#1-why-a-transformer)
2. [Input Features](#2-input-features)
3. [Sequence Construction](#3-sequence-construction)
4. [Model Architecture](#4-model-architecture)
5. [Training](#5-training)
6. [Inference: Return → Price](#6-inference-return--price)
7. [Long-Term Prediction Adjustment](#7-long-term-prediction-adjustment)
8. [Evaluation Metric](#8-evaluation-metric)
9. [Pre-trained Model Accuracies](#9-pre-trained-model-accuracies)
10. [Class Reference](#10-class-reference)

---

## 1. Why a Transformer?

Traditional approaches to time series prediction (RNNs, LSTMs) process sequences step by step and struggle to capture long-range dependencies — how something that happened 50 days ago relates to today's price.

Transformers use **self-attention**, which lets every position in the sequence directly attend to every other position regardless of distance. This makes them well-suited to financial data, where patterns like quarterly earnings cycles, seasonal trends, or recurring volatility regimes can span many time steps.

---

## 2. Input Features

Rather than feeding raw closing prices into the model, 7 derived features are computed from the price history. These give the model richer signal about momentum, trend, and risk:

| Feature | Formula | What it captures |
|---------|---------|-----------------|
| `Returns` | `(Close_t − Close_{t-1}) / Close_{t-1}` | Day-over-day percentage change |
| `MA_10` | Rolling 10-period mean of Close | Short-term trend direction |
| `MA_30` | Rolling 30-period mean of Close | Medium-term trend direction |
| `MACD` | `EMA(12) − EMA(26)` | Momentum — when short EMA crosses long EMA |
| `RSI` | 14-period Relative Strength Index | Overbought / oversold signal (0–100) |
| `Volatility_10` | Rolling 10-period std of returns × √10 | Recent short-term risk |
| `Volatility_30` | Rolling 30-period std of returns × √30 | Recent medium-term risk |

Using `Returns` (percentage changes) rather than raw prices makes the input **stationary** — the model is not thrown off by stocks that trade at very different absolute price levels (e.g. `NVDA` at $900 vs a stock at $10).

All 7 features are normalised to `[0, 1]` using a `MinMaxScaler` fitted on the training set. The fitted scaler is stored and reused at inference time.

---

## 3. Sequence Construction

The model uses a **sliding window** approach. Given a time series of scaled features, windows of length **60 time steps** are created. Each window is one training sample:

```
Time:  t=0    t=1    ...    t=59   |   t=60   ← label
       ──────────────────────────   |   ──────
       [60 × 7 feature matrix]      |   Returns value to predict
```

An 80% / 20% chronological train/test split is used. The data is **not shuffled** — future data must never appear in the training set.

---

## 4. Model Architecture

```
Input
  shape: [batch_size, 60, 7]
    │
    ▼
Linear Embedding
  nn.Linear(7 → 128)
  Projects the 7 input features into the model's
  internal d_model=128 dimensional space.
    │
    ▼
+ Positional Encoding
  nn.Parameter([1, 60, 128])  ← learned, not sinusoidal
  Added element-wise to the embedded input.
  Tells the model the relative position of each time step.
    │
    ▼
TransformerEncoder  (repeated × 4 layers)
  ┌────────────────────────────────────────────┐
  │  MultiHeadAttention                         │
  │    heads:   8                               │
  │    d_model: 128  →  head_dim: 16 each       │
  │                                             │
  │  FeedForward                                │
  │    hidden dim: 256                          │
  │    activation: ReLU (PyTorch default)       │
  │    dropout:    0.2                          │
  │                                             │
  │  LayerNorm + residual connections           │
  └────────────────────────────────────────────┘
    │
    ▼
Take last time step
  encoder_output[:, -1, :]
  shape: [batch_size, 128]
  Only the final position's representation is used
  for prediction (it has attended to all prior steps).
    │
    ▼
Output Head
  nn.Linear(128 → 1)
    │
    ▼
Output
  shape: [batch_size, 1]
  A single scaled return value — the predicted
  percentage change for the next time step.
```

**Why encoder-only (no decoder)?** The model predicts one return value at a time rather than a full output sequence, so a decoder stack is not needed. The encoder alone is sufficient to learn temporal dependencies across the 60-step context window.

---

## 5. Training

| Hyperparameter | Value |
|----------------|-------|
| Loss function | Mean Squared Error (MSE) |
| Optimiser | Adam |
| Learning rate | `0.0001` |
| Batch size | `32` |
| Epochs | `100` |
| Device | CUDA GPU if available, else CPU |

Training is handled by `ModelTrainer.train()`. Loss is printed every 5 epochs. The final checkpoint (model weights + optimiser state) is saved to `models/<TICKER>_model.pt`.

---

## 6. Inference: Return → Price

The model predicts a **scaled return** (percentage change), not a price directly. At inference, the return is first inverse-transformed back to an unscaled return, then compounded from the last known closing price:

```
price_{t+1} = price_t  × (1 + return_{t+1})
price_{t+2} = price_{t+1} × (1 + return_{t+2})
...
```

After each step, the newly predicted return is appended to the input sequence and the oldest step is dropped (rolling window), allowing the model to generate forecasts for any number of future steps.

---

## 7. Long-Term Prediction Adjustment

For predictions of **5 days or fewer**, the model's output is used directly.

For predictions **beyond 5 days**, the raw model output is blended with a random walk component. This reflects the real-world observation that forecasting models lose signal quality over longer horizons — beyond a few days, market prices are increasingly driven by unpredictable news and macro events.

### The Blending Formula

For each step `i` beyond the first 5:

```
combined_return = (model_return × model_weight)
                + (random_component × random_weight)

where:
  model_weight  = 1 − decay_rate[i]
  random_weight = decay_rate[i]
  decay_rate    = linspace(0 → 0.95, n_steps)   # for very long horizons
                  linspace(0 → 0.85, n_steps)   # for shorter long-term
```

The `random_component` is drawn from:

```
Normal(mean = historical_mean_daily_return × 0.7,
       std  = daily_volatility × 1.2)
```

The mean is slightly dampened (`× 0.7`) and the standard deviation slightly elevated (`× 1.2`) to produce conservative, realistic-looking uncertainty.

### Mean Reversion (after ~22 days)

After approximately one month of predictions, a mean reversion pull is applied. If the cumulative predicted return has drifted further than historical volatility would statistically expect over that timeframe:

```
if |total_return_so_far| > historical_volatility × √(days/252):
    reversion = -sign(total_return) × reversion_strength
    combined_return += reversion

where reversion_strength = 0.03 × (i / n_steps)   # grows with time
```

This prevents the model from producing unrealistically runaway predictions over multi-month horizons.

### Price Bounding

All predictions are clipped to a maximum range of `±2σ` of the statistically expected price movement, where `σ` scales with the square root of time (consistent with the random walk / efficient market assumption):

```
max_change = historical_annual_volatility × √(days/252) × 2
price[i]   = clip(price[i], current_price × (1 − max_change),
                             current_price × (1 + max_change))
```

---

## 8. Evaluation Metric

After training, the model is evaluated on the held-out test set. The reported accuracy is:

```
RMSE     = √( mean( (predicted_returns − actual_returns)² ) )
accuracy = 1 − (RMSE / mean(actual_returns))
```

This is a normalised error metric. An accuracy of `0.96` means the RMSE is 4% of the mean return value.

---

## 9. Pre-trained Model Accuracies

| Ticker | Exchange | Accuracy |
|--------|----------|----------|
| `AAPL` | NASDAQ | 96.3% |
| `NKE` | NYSE | 89.7% |
| `NVDA` | NASDAQ | 89.2% |
| `HDFCBANK.NS` | NSE India | 89.0% |

For any other ticker, a new model is trained automatically on first prediction request.

---

## 10. Class Reference

### `StockDataHandler`

Manages data for a single ticker.

| Method | What it does |
|--------|-------------|
| `fetch_data()` | Downloads full history from Yahoo Finance and saves to `data/<TICKER>.csv` |
| `update_data()` | Appends only the days missing since the last saved date |
| `get_data()` | Loads from CSV or falls back to `fetch_data()` |
| `calculate_technical_indicators(df)` | Adds MA_10, MA_30, MACD, RSI, Bollinger Bands columns |
| `calculate_volatility(df)` | Adds Volatility_10 and Volatility_30 columns |
| `prepare_data()` | Runs indicators + volatility, computes Returns, scales features, builds `[X, y]` sequences, splits train/test |

---

### `StockDataset`

A thin PyTorch `Dataset` wrapper. Converts numpy arrays to `FloatTensor` and implements `__len__` / `__getitem__` so it works directly with `DataLoader`.

---

### `StockTransformer`

The neural network. Parameters:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `input_dim` | 7 | Number of input features |
| `d_model` | 128 | Internal embedding dimension |
| `nhead` | 8 | Number of attention heads |
| `num_encoder_layers` | 4 | Number of stacked encoder blocks |
| `dim_feedforward` | 256 | Hidden size of the feedforward sublayer |
| `dropout` | 0.2 | Dropout probability |
| `seq_len` | 60 | Input sequence length |

---

### `ModelTrainer`

Handles training, evaluation, saving, and loading.

| Method | What it does |
|--------|-------------|
| `create_model()` | Instantiates `StockTransformer` with config values and moves to device |
| `train(dataloader)` | Runs the training loop for `epochs` iterations using MSE + Adam |
| `evaluate(dataloader)` | Computes MSE, MAE, RMSE, and accuracy on the test set |
| `predict_next_n(input_seq, n_steps)` | Autoregressively predicts `n_steps` future returns, converts to prices |
| `save_model(path)` | Saves model + optimiser state dict to `.pt` file |
| `load_model(path)` | Loads checkpoint or creates a fresh model if file not found |

---

### `StockManager`

High-level orchestrator that ties everything together.

| Method | What it does |
|--------|-------------|
| `add_stock(ticker)` | Downloads data and registers the ticker in `config.json` |
| `train_stock(ticker)` | Runs full train pipeline and updates accuracy in config |
| `evaluate_stock(ticker)` | Re-evaluates a trained model on current test data |
| `predict_stock(ticker, duration)` | Loads model, generates predictions, applies long-term adjustments |
| `update_all_stocks()` | Refreshes data for all registered tickers; retrains if accuracy < 0.7 |
