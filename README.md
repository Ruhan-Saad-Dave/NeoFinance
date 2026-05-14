<div align="center">

# NeoFinance

### AI-Powered Financial Assistant with Agentic Architecture

*Built in 24 hours at a hackathon*

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Custom%20Transformer-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini%201.5%20Pro-Function%20Calling-4285F4?style=for-the-badge&logo=google&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-Web%20UI-FF7C00?style=for-the-badge)

</div>

---

## Overview

NeoFinance is a conversational AI financial assistant that demonstrates an end-to-end **agentic AI system** built from scratch. Users interact through a natural language chat interface — the LLM autonomously decides which tools to invoke, executes them, and synthesises the results into a coherent response. No explicit routing logic. No hardcoded workflows.

The system integrates a **custom-built PyTorch Transformer** for multi-step stock price forecasting, real-time financial data from multiple APIs, voice input via Whisper, and multilingual support — all orchestrated through a single agentic loop.

---

## Agentic Architecture

The core of NeoFinance is a **function-calling agentic loop** powered by Google Gemini 1.5 Pro. Fifteen Python tools — covering stock data, ML predictions, forex, crypto, loans, news sentiment, and more — are exposed to the LLM as callable functions. Gemini reads their names, descriptions, and type signatures, then decides autonomously which tool to call, with what arguments, and whether to chain multiple calls together.

```
User Input (text or voice)
        ↓
  Gemini 1.5 Pro
  ┌─ reads tool schemas ──────────────────────────────────┐
  │  selects tool(s), determines arguments                 │
  │  chains calls if needed                                │
  └───────────────────────────────────────────────────────┘
        ↓
  Python Tool Execution
  (live data fetch / ML inference)
        ↓
  Gemini formats results → streamed back to UI
```

Each user session maintains its own isolated Gemini chat context, ensuring conversation history is never shared across concurrent users.

---

## Custom Transformer Model

A Transformer neural network was designed and implemented from scratch in PyTorch for multi-step stock price forecasting — no pre-built forecasting library used.

### Feature Engineering

Rather than using raw prices, the model is trained on **7 derived financial features** computed from historical closing prices:

| Feature | Description |
|---------|-------------|
| Daily Returns | Percentage change — makes the input stationary across stocks |
| MA-10 / MA-30 | Short and medium-term moving averages — trend direction |
| MACD | EMA(12) − EMA(26) — momentum signal |
| RSI (14-period) | Relative Strength Index — overbought / oversold signal |
| Volatility-10 / Volatility-30 | Rolling annualised volatility — market risk signal |

All features are normalised to [0, 1] using a fitted MinMaxScaler stored alongside each model.

### Architecture

| Component | Detail |
|-----------|--------|
| Input | Sliding window of 60 time steps × 7 features |
| Embedding | Linear projection 7 → 128 (d_model) |
| Positional Encoding | Learned parameters (not sinusoidal) |
| Encoder | 4 layers, 8-head attention, 256-dim feedforward, 0.2 dropout |
| Output Head | Linear 128 → 1 (predicts next-step return) |
| Inference | Autoregressive — output is fed back as input for multi-step forecasting |

The model predicts **returns** (percentage changes), not raw prices. Predictions are converted back to prices by compounding from the last known close. This makes the architecture stock-agnostic — the same model structure works across any ticker regardless of price magnitude.

### Long-Term Uncertainty Modelling

For predictions beyond 5 days, the model's output is blended with a historically-calibrated random walk component. The model's weight decays linearly with the forecast horizon, while the random component scales with `√t` — consistent with the efficient market hypothesis. A mean-reversion pull is applied after ~22 days to prevent runaway long-horizon drift. All predictions are bounded within a `±2σ` confidence envelope.

### Accuracy on Pre-trained Models

| Ticker | Exchange | Test Accuracy |
|--------|----------|---------------|
| AAPL | NASDAQ | 96.3% |
| NKE | NYSE | 89.7% |
| NVDA | NASDAQ | 89.2% |
| HDFCBANK.NS | NSE India | 89.0% |

For any untrained ticker, the system automatically fetches data, trains a model, and saves it for future use.

---

## Capabilities

| Category | What it can do |
|----------|---------------|
| **Stock Data** | Historical daily prices, intraday 1-min OHLCV, company financial overview |
| **Prediction** | Multi-day price forecast (daily model), intraday forecast (intraday model), with confidence interval charts |
| **Market** | Top 10 daily gainers, losers, most actively traded |
| **Forex & Crypto** | Live and historical exchange rates for fiat and cryptocurrency pairs |
| **Commodities** | Historical prices for Gold, Silver, Platinum, Copper |
| **News** | Top 10 news articles with sentiment labels (Bullish / Bearish / Neutral) per stock |
| **Finance** | Loan and mortgage calculators with full amortisation schedule |
| **Voice** | Speech-to-text via Whisper — voice queries processed identically to text |
| **Multilingual** | Detects user language, translates internally to call the right tool, responds in the user's language |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM & Agentic Orchestration | Google Gemini 1.5 Pro via `google-genai` SDK |
| ML Framework | PyTorch — custom Transformer architecture |
| Feature Engineering | pandas, NumPy, scikit-learn (MinMaxScaler) |
| Speech Recognition | Hugging Face Transformers — OpenAI Whisper (whisper-small) |
| Web UI & Streaming | Gradio (per-session state, streaming responses) |
| Stock Data | yfinance (Yahoo Finance) |
| Financial Data & News | Alpha Vantage API |
| Chart Hosting | Cloudinary |
| Visualisation | Matplotlib |

---

## Setup

### 1. Clone & Install

```bash
git clone https://github.com/Ruhan-Saad-Dave/Neo.git
cd Neo
python -m venv myenv

# Windows
myenv\Scripts\activate

# macOS / Linux
source myenv/bin/activate

pip install -r requirements.txt
```

### 2. API Keys

Create a `.env` file in the project root:

```
GEMINI_API_KEY="..."
ALPHA_API_KEY="..."
CLOUDINARY_CLOUD_NAME="..."
CLOUDINARY_API_KEY="..."
CLOUDINARY_API_SECRET="..."
```

| Key | Where to get it |
|-----|----------------|
| `GEMINI_API_KEY` | [Google AI Studio](https://makersuite.google.com/app/apikey) |
| `ALPHA_API_KEY` | [Alpha Vantage](https://www.alphavantage.co/support/#api-key) — free tier available |
| Cloudinary keys | [Cloudinary Console](https://console.cloudinary.com/pm) → Settings → API Keys |

> **Note:** The Alpha Vantage free tier allows 25 requests per day. GPU is recommended for training new models; CPU training can take up to 2 hours.

### 3. Run

```bash
# Windows
python app.py

# macOS / Linux
python3 app.py
```

Gradio will print a local URL and a temporary public share link.

---

## Documentation

Full technical documentation is in the [`docs/`](docs/) folder:

- [System Overview & Agentic Architecture](docs/system_overview.md)
- [Tool Reference — all 15 tools explained](docs/tools.md)
- [Transformer Architecture — full technical deep-dive](docs/transformer_architecture.md)

---

## Contributors

Built by **Team Code Crusaders** at a 24-hour hackathon.

| Contributor | Role |
|-------------|------|
| **Hano Varghese** | Transformer architecture, model training (daily & intraday), prediction charts, stock info retrieval |
| **Ruhan Dave** | Gemini API integration & function calling, speech-to-text, multilingual support, forex, crypto, and commodity data |

---

## Putting This on Your Resume

### Header Line

```
NeoFinance — Agentic AI Financial Assistant          GitHub: github.com/Ruhan-Saad-Dave/Neo
Tech: Python · PyTorch · Google Gemini · Gradio · Hugging Face Transformers · Alpha Vantage
```

### Bullet Points

Pick 3–4 based on the job description. Each is written in resume style: **what you did → how → result/scale**.

**Agentic AI — lead with this, it's the most differentiated skill:**
> Designed and implemented an agentic AI system using Google Gemini 1.5 Pro function calling, exposing 15 domain-specific financial tools that the LLM selects and orchestrates autonomously at runtime

**Custom Transformer — shows ML depth, not just API usage:**
> Built a stock price forecasting Transformer from scratch in PyTorch — custom multi-head attention encoder, learned positional encoding, and autoregressive multi-step inference — achieving 96.3% test accuracy on AAPL

**Feature engineering + uncertainty modelling — shows you understand the ML, not just the framework:**
> Engineered 7 technical indicator features (MACD, RSI, Bollinger Bands, rolling volatility) and implemented a long-horizon uncertainty model blending transformer output with historically-calibrated random walk and mean reversion

**Multimodal + multilingual — shows breadth:**
> Integrated OpenAI Whisper for speech-to-text input and multilingual support — the LLM translates queries internally before tool dispatch and responds in the user's language

**Shipped under pressure — hackathon context:**
> Delivered end-to-end in 24 hours at a hackathon, covering data pipelines (Yahoo Finance, Alpha Vantage), real-time streaming UI (Gradio), and cloud image hosting (Cloudinary) for prediction charts

### Picking the Right Bullets

| Job description emphasises... | Prioritise |
|-------------------------------|-----------|
| "LLM agents", "tool use", "agentic systems" | Agentic AI bullet |
| "ML engineering", "model development" | Transformer + feature engineering bullets |
| "full-stack AI", "end-to-end ML" | Hackathon + agentic AI bullets |
| "NLP", "multimodal" | Multimodal bullet |

### Skills Section Keywords

Make sure these appear somewhere in your Skills section — ATS systems and engineers both scan for them:

```
PyTorch · Transformer Architecture · LLM Function Calling · Agentic AI ·
Hugging Face Transformers · Python · Time Series Forecasting · Gradio ·
Google Gemini API · REST APIs · Feature Engineering
```

### One-Liner

For tight resumes, a LinkedIn featured project, or as a talking point in interviews:

> Built an agentic AI financial assistant with a custom PyTorch Transformer (96.3% accuracy) and 15 LLM-callable tools, shipped in 24 hours at a hackathon
