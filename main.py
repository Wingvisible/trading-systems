import yfinance as yf
import feature_vectors
import numpy as np
import k_means
import visualisation
import pickle
from sklearn.preprocessing import StandardScaler

ticker = "^GSPC"
df = yf.download(
    ticker,
    start="2000-01-01",
    end="2025-01-01",
    interval="1d"
)
# open  high  low  close  volume  dividends  stock-splits

close = df[["Close"]].copy()
close.dropna(inplace=True)

train_prices = close.loc["2000-01-01":"2018-01-01"].values
data = feature_vectors.generate_data_matrix(train_prices)
np.save('data_matrix.npy', data)

X = np.load("data_matrix.npy")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
T = 100000
centroids = k_means.K_means_algorithm(X_scaled, T, 3)

data_length = len(train_prices[50:])
x_length = X_scaled.shape[0]
print(data_length, x_length)
visualisation.plot(train_prices[50:], X_scaled, centroids)

np.save('centroids.npy', centroids)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)




