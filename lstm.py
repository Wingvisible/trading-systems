import torch, pickle
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
from k_means import centroid_probabilities
import yfinance as yf



# 5 normalised returns: return / (std_t * sqrt(h)) for h = 1,21,63,126,252
# 3 MACD signals:  MACD(S,L) / std_p_63  / std(MACD over 1yr) for (S,L) = (8,24),(16,48),(32,96)
# sigma_t = EWM std of daily returns with 60-day span 
def build_features(prices):
    s = pd.Series(prices.squeeze())
    #log returns
    returns = np.log(s).diff().fillna(0)
    #take the exponentially weigthed moving standard deviation
    sigma_t = returns.ewm(span=60, adjust=False).std(bias=True).clip(lower=1e-8)
    features = pd.DataFrame(index=s.index)

    #log returns of t-window to t
    cumulative_returns = returns.cumsum()
    for i, window in enumerate([1, 21, 63, 126, 252]):
        window_sum = cumulative_returns - cumulative_returns.shift(window).fillna(0)
        features[i]   = window_sum / (sigma_t * window**0.5)
    
    #macd indicators
    for i, (S, L) in enumerate([(8, 24), (16, 48), (32, 96)]):
        macd = s.ewm(span=S, adjust=False).mean() - s.ewm(span=L, adjust=False).mean()
        std_p = s.rolling(64,  min_periods=1).std(ddof=0).clip(lower=1e-8)
        q = macd / std_p
        std_q = q.rolling(253, min_periods=1).std(ddof=0).clip(lower=1e-8)
        Y = q / std_q
        features[5+i] = Y
    
    #k_means centroid probabilities with k = 3 for log momentum and standard deviation of windows 5, 10, 20, 30, 40, 50 days
    matrix_column_list = []
    for j in [5,10, 20, 30, 40, 50]:
        momentum = (cumulative_returns - cumulative_returns.shift(j).fillna(0)).values
        risk = (returns.rolling(j, min_periods=1).std(ddof=0).clip(lower=1e-8)).values
        matrix_column_list.append(momentum)
        matrix_column_list.append(risk)
    X = np.column_stack(matrix_column_list)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    X_scaled = scaler.transform(X)
    centroids = np.load("centroids.npy")
    probabilities = centroid_probabilities(X=X_scaled, centroids=centroids)
    all_features = np.concatenate([features.values.astype(np.float32), probabilities.astype(np.float32)], axis=1) 
    
    forward_returns = returns.shift(-1).fillna(0)

    return (all_features,
            sigma_t.values.astype(np.float32),
            forward_returns.values.astype(np.float32))

# dataset class for the prices
class PriceDataset(Dataset):
    def __init__(self, features, sigma, forward_returns, seq_len=63, stride = 20):
        self.features   = torch.tensor(features)
        self.sigma = torch.tensor(sigma)
        self.returns = torch.tensor(forward_returns)
        self.length   = seq_len #sequence of 63
        self.sequences = list(range(0, len(self.features) - self.length, stride))

    def __len__(self):  #number of sequences
        return len(self.sequences)

    def __getitem__(self, i): #get a specific sequence
        sequence = slice(self.sequences[i], self.sequences[i] + self.length)
        return self.features[sequence], self.sigma[sequence], self.returns[sequence]


# lstm model
class LSTM(nn.Module):
    def __init__(self, input_size=11, hidden_size=64, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):   # x: (B, 63, 11)
        h, _ = self.lstm(x)   # h: (B, 63, 64)
        return torch.tanh(self.head(self.drop(h))).squeeze(-1)  # (B, 63)



# Equation 35
# R_t = position_t * (sigma_tgt/sigma_t) * forward_return  −  c * | position_t/sigma_t − positioin_t−1/sigma_t−1 | * σ_tgt
# Sharpe loss Equation 16
# Loss = − mean(R) * √252 / std(R)
def sharpe_loss(positions, sigma, forward_return, cost_bps=0.0, sigma_tgt=0.15):
    c = cost_bps / 10_000
    #A = position_t * (sigma_tgt/sigma_t) * forward_return
    w = positions / sigma.clamp(min=1e-8)               
    A = w * sigma_tgt * forward_return #shape (B, 63)                    

    #B = | position_t/sigma_t − positioin_t−1/sigma_t−1 |
    B = torch.diff(w, dim=1) #shape (B, 62)                           
    turnover = c * sigma_tgt * B.abs() 
    turnover = torch.cat([torch.zeros_like(w[:, :1]), turnover], dim=1) #add a 0 in front to get shape (B, 63) 

    R  = (A - turnover).reshape(-1) #shape (B * 63,)                         
    return -(R.mean() * 252**0.5 / R.std().clamp(min=1e-8)) # negative Sharpe

#train function
def train(model, loader, epochs=10, lr=1e-3, cost_bps=0.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        total = 0
        model.train()
        for features, sigma, forward_returns in loader:
            features, sigma, forward_returns = features.to("mps"), sigma.to("mps"), forward_returns.to("mps")
            optimizer.zero_grad()
            loss = sharpe_loss(model(features), sigma, forward_returns, cost_bps=cost_bps)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        print(f"epoch {epoch:3d}  loss={total/len(loader):+.4f}")

def evaluate(model, loader, cost_bps=10.0):
    model.eval()
    all_positions, all_sigma, all_returns = [], [], []    
    with torch.no_grad():
        for features, sigma, forward_returns in loader:
            positions = model(features.to("mps"))
            all_positions.append(positions.cpu())
            all_sigma.append(sigma)
            all_returns.append(forward_returns)
    
    positions = torch.cat(all_positions)   # (N, 63) where N is number of sequences
    sigma     = torch.cat(all_sigma)       # (N, 63)
    returns   = torch.cat(all_returns)     # (N, 63)

    sharpe = -sharpe_loss(positions, sigma, returns, cost_bps=cost_bps).item()
    return sharpe

if __name__ == "__main__":
    
    ticker = "^GSPC"
    df = yf.download(ticker, start="2000-01-01", end="2025-01-01", interval="1d")
    close = df[["Close"]].dropna()

    #TRAIN/TEST 
    train_prices = close.loc["2000-01-01":"2018-01-01"].values
    test_prices  = close.loc["2018-01-02":"2025-01-01"].values

    #BUILD FEATURES
    train_feats, train_sig, train_ret = build_features(train_prices)
    test_feats,  test_sig,  test_ret  = build_features(test_prices)

    #DATASETS
    # Train: stride=20 (overlap for more data)
    train_dataset = PriceDataset(train_feats, train_sig, train_ret, seq_len=63, stride = 20)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    # Test: stride=63 (no overlap)
    test_dataset  = PriceDataset(test_feats, test_sig, test_ret, seq_len=63, stride = 63)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=False)
    
    #initialize model
    model = LSTM(input_size=11, hidden_size=65, dropout=0.5).to("mps")

    #TRAIN
    print("── Training ──")
    train(model, train_loader, epochs=50, cost_bps=10)

    #TEST
    print("\n── Out-of-Sample Test ──")
    sharpe_0 = evaluate(model, test_loader, cost_bps=0.0)
    sharpe_10 = evaluate(model, test_loader, cost_bps=10.0)
    sharpe_50 = evaluate(model, test_loader, cost_bps=50.0)

    print(f"  0 bps:  {sharpe_0:.3f}")
    print(f"  10 bps: {sharpe_10:.3f}")
    print(f"  50 bps: {sharpe_50:.3f}")