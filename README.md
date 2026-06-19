# Nifty 50 Market Direction Prediction Model

A machine learning model that predicts the **next-day directional movement** of the Nifty 50 index using a Random Forest classifier trained on 15+ years of historical price data and engineered momentum/trend features.

---

## What This Does

This isn't trying to predict exact prices — that's a much harder (often unsolvable) problem. Instead, it tackles a simpler, more tractable question: **will tomorrow's close be higher than today's close?** A binary up/down signal.

The model is trained on the full history of the Nifty 50 and validated using a **rolling-window backtest** — meaning it's only ever tested on data it hasn't seen, retrained periodically as if running live, exactly how a real systematic strategy would be evaluated.

---

## Key Result

- **56.7% precision** on positive (buy) signals across 3,000+ out-of-sample data points
- This may sound modest, but it's a **statistically meaningful edge** above the 50% random baseline — in systematic trading, edges this size compound meaningfully at scale
- The model's performance is strongest during trending markets and weakens during high-volatility, mean-reverting regimes — an important limitation, documented below

---

## How It Works

### 1. Baseline Model
A first-pass Random Forest using only raw OHLCV data (`Close`, `Volume`, `Open`, `High`, `Low`) as predictors — establishing a baseline precision score before feature engineering.

### 2. Feature Engineering
The real value of this project is in the engineered features — built across **5 time horizons (2, 5, 60, 250, 1000 days)**:

- **Close Ratio**: `Close price ÷ rolling average` — captures whether price is above or below its recent trend
- **Trend**: sum of "up days" in the trailing window — captures recent momentum

These two feature types, repeated across short (2-day) to long (1000-day) horizons, let the model see both short-term noise and long-term structural trend simultaneously.

### 3. Rolling-Window Backtest
```
For each test window:
  1. Train on all data up to point i
  2. Predict on the next 250 days
  3. Move forward and repeat
```
This simulates how the model would actually be retrained and deployed over time — far more realistic than a single train/test split.

### 4. Probability Threshold Tuning
Instead of a default 50% classification threshold, the model only signals "Buy" when its predicted probability exceeds **60%** — trading off the number of signals generated for higher precision per signal.

---

## How to Run

**1. Install dependencies**
```bash
pip install yfinance pandas numpy matplotlib seaborn scikit-learn
```

**2. Run the script**
```bash
python Nifty_Signal.py
```

The script automatically downloads the full historical Nifty 50 dataset (`^NSEI`) via yfinance — no manual input required.

---

## Example Output

```
Predictions
0.0    1847
1.0    1153
Name: count, dtype: int64

Precision Score: 0.567
```

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| `yfinance` | Historical Nifty 50 OHLCV data |
| `pandas` | Feature engineering, rolling windows |
| `numpy` | Numerical operations |
| `scikit-learn` | RandomForestClassifier, precision_score |
| `matplotlib` / `seaborn` | Visualisation |

---

## What I Learned / Key Insight

The single biggest improvement came not from tuning the model, but from **feature engineering across multiple time horizons**. The baseline model (raw OHLCV only) performed close to random. Adding the 5-horizon trend and ratio features was what pushed precision meaningfully above 50%.

Equally important: the model's edge is **regime-dependent**. It performs best in trending markets and degrades in choppy, mean-reverting conditions — a finding that matters more than the headline precision number, because it tells you *when* to trust the signal and when not to.

---

## Limitations

- Precision >50% does not guarantee profitability — transaction costs, slippage, and position sizing all matter
- The model has not been deployed live or tested with real capital
- Markets are non-stationary — relationships that held in the training period may not hold going forward

---

*This is not financial advice. Built for educational and research purposes.*
