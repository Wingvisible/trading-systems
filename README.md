# S&P 500 Regime-Aware LSTM Trading Model

An LSTM that learns to trade the S&P 500 (`^GSPC`) by directly maximizing the Sharpe ratio. K-means market-regime probabilities are added to momentum and trend features as inputs.

**Result:** trained on 2000–2018 and tested out-of-sample on 2018–2025, the model achieved a **Sharpe ratio of 0.515**.

## How It Works

1. **Regime detection (`main.py`, `feature_vectors.py`, `k_means.py`)**
   - Downloads daily S&P 500 closes (2000–2025) with `yfinance`.
   - For windows of 5, 10, 20, 30, 40 and 50 days, computes log momentum and rolling volatility, giving a 12-dimensional vector per day.
   - Standardizes the vectors and clusters them with a **from-scratch NumPy K-means** (K = 3, vectorized distance computation) to find market regimes.
   - Saves the fitted `scaler.pkl` and `centroids.npy`, both fit on training data only.

2. **Features (`lstm.py`)** – 11 inputs per day:
   - 5 volatility-normalized returns (1, 21, 63, 126, 252-day horizons)
   - 3 normalized MACD signals, for (8, 24), (16, 48) and (32, 96) spans
   - 3 regime probabilities, a softmax over negative squared distances to the K-means centroids

3. **Model (`lstm.py`)**
   - Single-layer LSTM (hidden size 65, dropout 0.5) with a linear head and `tanh`, so the output is a position in [-1, 1].
   - Input sequences are 63 trading days long.

4. **Loss**
   - Negative annualized **Sharpe ratio** of volatility-targeted returns (15% target vol), net of transaction costs on turnover.
   - Training uses 10 bps costs.

## Inspiration

This project combines ideas from two papers:

- **Lim, Zohren & Roberts (2019), [*Enhancing Time Series Momentum Strategies Using Deep Neural Networks*](https://arxiv.org/abs/1904.04912).** Source of the LSTM that outputs positions directly, the Sharpe-ratio loss, volatility scaling (15% target, 60-day EWM std), the normalized-return and MACD input features, and the turnover-cost term in the loss. I adapted it to a single asset (the S&P 500) instead of their 88-futures portfolio.
- **Oliva & Tinjala (2025), [*Modeling Market States with Clustering and State Machines*](https://arxiv.org/abs/2510.00953).** Source of the momentum and risk feature vectors (5–50 day windows) clustered with K-means to find market regimes. The paper uses the clusters to build a state machine and a return-distribution model.

**My contribution:** I merged the two. The paper's K-means regimes are not used as a standalone model. Instead, each day's soft cluster-membership probabilities become extra input features for the Sharpe-optimized LSTM. The LSTM can then condition its positions on the current market regime.

## Setup

| | |
|---|---|
| Train period | 2000-01-01 to 2018-01-01 |
| Test period | 2018-01-02 to 2025-01-01 |
| Train sequences | 63 days, stride 20 (overlapping) |
| Test sequences | 63 days, stride 63 (non-overlapping) |
| Optimizer | Adam, lr 1e-3, grad-clip 1.0, 50 epochs, batch size 32 |

## Usage

```bash
pip install numpy pandas torch scikit-learn yfinance tqdm matplotlib

python main.py   # fits scaler + K-means regimes, saves scaler.pkl and centroids.npy
python lstm.py   # builds features, trains the LSTM, prints out-of-sample Sharpe
```

`lstm.py` uses the Apple `mps` device. Replace `"mps"` with `"cuda"` or `"cpu"` on other hardware.

## Files

| File | Purpose |
|---|---|
| `main.py` | Data download, scaling, K-means fitting, regime plot |
| `feature_vectors.py` | Momentum and risk feature matrix for clustering |
| `k_means.py` | K-means and centroid-probability functions |
| `lstm.py` | Feature builder, dataset, LSTM, Sharpe loss, train/evaluate |

# Context
Other regime detection methods were also experimented to see difference with K-means.
