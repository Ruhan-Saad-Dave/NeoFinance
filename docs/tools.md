# Tool Reference

All tools are defined in `tools.py` and registered with Gemini in `app.py`. Gemini reads each function's name, docstring, and type annotations to understand what each tool does and when to call it.

For how these tools fit into the broader system, see [system_overview.md](system_overview.md).

---

## Table of Contents

| Tool | Category | Data Source |
|------|----------|-------------|
| [`fetch_stock_info`](#fetch_stock_infosymbol-str) | Stock | Alpha Vantage |
| [`fetch_stock_history`](#fetch_stock_historyticker-str-days-int) | Stock | Yahoo Finance |
| [`predict_stock_price`](#predict_stock_priceticker-str-days-int) | Prediction | Yahoo Finance + local model |
| [`predict_stock_intraday_price`](#predict_stock_intraday_priceticker-str-mins-int) | Prediction | Alpha Vantage + local model |
| [`currency_exchange`](#currency_exchangefrom_name-str-to_name-str) | Forex / Crypto | Alpha Vantage |
| [`currency_exchange_daily`](#currency_exchange_dailyfrom_name-str-to_name-str-days-int) | Forex | Alpha Vantage |
| [`crypto_currency_exchange`](#crypto_currency_exchangefrom_name-str-to_name-str) | Crypto | Alpha Vantage |
| [`digital_currency_daily`](#digital_currency_dailyfrom_name-str-to_name-str-days-int) | Crypto | Alpha Vantage |
| [`calculation`](#calculationnum1-float-num2-float-mode-str) | Utility | — |
| [`loan_calculator`](#loan_calculatorprincipal-float-annual_rate-float-years-int) | Finance | — |
| [`mortgage_calculator`](#mortgage_calculatorprincipal-float-annual_rate-float-years-int) | Finance | — |
| [`top_gainers_losers`](#top_gainers_losers) | Market | Alpha Vantage |
| [`fetch_commodity_history`](#fetch_commodity_historyticker-str-days-int) | Commodity | Yahoo Finance |
| [`get_news_sentiment`](#get_news_sentimentticker-str-topic-str) | News | Alpha Vantage |
| [`get_intraday_stock`](#get_intraday_stockticker-str-mins-int) | Stock | Alpha Vantage |

---

## Stock & Company Data

### `fetch_stock_info(symbol: str)`

Returns a financial analysis of a company — market capitalisation, annual revenue, and revenue multiple — with plain-English interpretation of each figure.

**Data source:** Alpha Vantage `OVERVIEW` endpoint.

**Ticker format:**
- US stocks: standard symbol — `AAPL`, `MSFT`
- Indian NSE: append `.NS` — `RELIANCE.NS`
- Indian BSE: append `.BO` — `HDFCBANK.BO`

**Sample output:**
```
💰 Market Capitalisation: $3,000,000,000,000
📈 Total Revenue: $400,000,000,000
📊 Revenue Multiple: 7.50
```

---

### `fetch_stock_history(ticker: str, days: int)`

Returns the daily closing price of a stock for the past `days` calendar days.

**Data source:** Yahoo Finance.

**Ticker format:** Same as above; additionally `.L` for London Stock Exchange.

**Sample output:**
```
📅 Date: 2025-04-01, Close: $172.43
📅 Date: 2025-04-02, Close: $174.10
...
```

---

### `get_intraday_stock(ticker: str, mins: int)`

Returns the last `mins` minutes of 1-minute interval OHLCV (Open, High, Low, Close, Volume) data for a stock. Useful for viewing very recent price action.

**Data source:** Alpha Vantage `TIME_SERIES_INTRADAY` at 1-minute resolution.

**Each record contains:** timestamp, open, high, low, close, volume.

---

## Stock Price Prediction

### `predict_stock_price(ticker: str, days: int)`

Predicts the daily closing price of a stock `days` into the future using a custom transformer model trained on Yahoo Finance data.

**Use this tool when the time duration is in days, weeks, months, or years.**

If no pre-trained model exists for `ticker`, the system will:
1. Download full historical data from Yahoo Finance
2. Train a new transformer model (~2 hours on CPU, faster on GPU)
3. Save the model to `models/<TICKER>_model.pt`

**Output includes:** Predicted price, current price, expected % change, model accuracy, and a link to a Cloudinary-hosted prediction chart.

For the model's inner workings, see [transformer_architecture.md](transformer_architecture.md).

---

### `predict_stock_intraday_price(ticker: str, mins: int)`

Predicts the intraday stock price `mins` minutes into the future using a transformer trained on 1-minute interval data from Alpha Vantage.

**Use this tool when the time duration is in minutes or hours.**

Same auto-train behaviour as `predict_stock_price`. Models saved as `models/<TICKER>_model_intraday.pt`.

---

## Currency & Cryptocurrency

### `currency_exchange(from_name: str, to_name: str)`

Returns the current real-time exchange rate between two currencies. Works for both fiat and crypto pairs.

**Data source:** Alpha Vantage `CURRENCY_EXCHANGE_RATE`.

**Examples:** `USD`→`INR`, `EUR`→`JPY`, `BTC`→`USD`

**Returns:** Exchange rate and the timestamp of last refresh.

---

### `currency_exchange_daily(from_name: str, to_name: str, days: int)`

Returns the daily exchange rate between two fiat currencies for the past 7 available trading days.

**Data source:** Alpha Vantage `FX_DAILY`.

> Note: the `days` parameter is accepted but Alpha Vantage always returns up to 7 daily records regardless.

---

### `crypto_currency_exchange(from_name: str, to_name: str)`

Returns the current real-time exchange rate from a cryptocurrency to a fiat (or any currency pair involving crypto).

Functionally identical to `currency_exchange` — the two tools exist separately so Gemini reliably picks the right one based on the user saying "crypto rate" vs "currency rate".

---

### `digital_currency_daily(from_name: str, to_name: str, days: int)`

Returns historical daily prices for a cryptocurrency in a given fiat currency for the past 7 days.

**Data source:** Alpha Vantage `DIGITAL_CURRENCY_DAILY`.

**Fallback:** If this endpoint returns an error, Gemini is instructed to use `currency_exchange_daily` combined with `calculation` to approximate the result.

---

## Financial Calculators

### `calculation(num1: float, num2: float, mode: str)`

Performs basic arithmetic. Used by Gemini internally when it needs to combine or convert values from other tool results.

| Mode | Operation | Notes |
|------|-----------|-------|
| `addition` | `num1 + num2` | |
| `subtraction` | `num1 - num2` | |
| `multiplication` | `num1 * num2` | |
| `division` | `num1 / num2` | Returns error string if `num2 == 0` |
| `floor` | `num1 // num2` | Returns error string if `num2 == 0` |
| `remainder` | `num1 % num2` | Returns error string if `num2 == 0` |

---

### `loan_calculator(principal: float, annual_rate: float, years: int)`

Calculates the monthly payment, total cost, and total interest for a general loan — personal, auto, or student. **Not for home mortgages; use `mortgage_calculator` for those.**

Uses the standard loan amortisation formula:

```
Monthly Payment = P × r(1+r)^n / ((1+r)^n − 1)

where:
  P = principal
  r = annual_rate / 12 / 100   (monthly interest rate)
  n = years × 12               (total monthly payments)
```

**Returns:**
```json
{
  "Monthly Payment": 1234.56,
  "Total Cost": 148147.20,
  "Total Interest Paid": 48147.20
}
```

---

### `mortgage_calculator(principal: float, annual_rate: float, years: int)`

Calculates monthly payment, total cost, total interest, **and a full month-by-month amortisation schedule** for a home mortgage.

Uses the same formula as `loan_calculator` but additionally tracks the breakdown of each payment into its interest and principal components, and the remaining balance after each payment.

**Returns:** Payment summary plus an `"Amortization Schedule"` list with one entry per month:
```json
{
  "Month": 1,
  "Interest Paid": 833.33,
  "Principal Paid": 401.23,
  "Remaining Balance": 199598.77
}
```

---

## Market Data

### `top_gainers_losers()`

Returns the top 10 gaining stocks, top 10 losing stocks, and top 10 most actively traded stocks for the current trading day.

**Data source:** Alpha Vantage `TOP_GAINERS_LOSERS`.

**Each entry includes:** Ticker symbol, price, percentage change, trading volume.

> Note: Alpha Vantage free tier is limited to 25 requests/day. This call counts as one request.

---

### `fetch_commodity_history(ticker: str, days: int)`

Returns historical OHLCV price data for a commodity for the past `days` days.

**Data source:** Yahoo Finance using commodity futures tickers.

**Supported commodities:**

| Commodity | Ticker to pass |
|-----------|----------------|
| Gold | `GC=F` |
| Silver | `SI=F` |
| Platinum | `PL=F` |
| Copper | `HG=F` |

---

## News & Sentiment

### `get_news_sentiment(ticker: str, topic: str)`

Returns the 10 most recent news articles related to a stock and topic, including a sentiment label (Bullish / Bearish / Neutral) for each article and for each ticker mentioned within it.

**Data source:** Alpha Vantage `NEWS_SENTIMENT`.

**Available topics:**

```
blockchain, manufacturing, real_estate, retail_wholesale, technology,
earnings, ipo, mergers_and_acquisitions, financial_markets,
economic_monetary, economic_macro, life_sciences, economy_fiscal,
energy_transportation, finance
```

**Each result includes:** Title, URL, publication date/time, article summary, source name, overall sentiment, per-ticker sentiment breakdown.
