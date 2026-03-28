import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from pomegranate.hmm import DenseHMM
from pomegranate.distributions import StudentT

data = yf.download("^GSPC", start="2013-01-01", end="2023-01-01")

data["Returns"] = np.log(data["Close"] / data["Close"].shift(1))
data = data.dropna()
X = data["Returns"].values.reshape(-1, 1)

dists = [
    StudentT(dofs=5),
    StudentT(dofs=5)
]

model = DenseHMM(distributions=dists)

model.fit([X])

states = model.predict([X])[0]
data["Regime"] = states

state_counts = data["Regime"].value_counts()
state_percentages = state_counts / len(data) * 100

print("Number of days in each state:")
print(state_counts)

print("\nPercentage of time in each state:")
print(state_percentages)

means = []
variances = []
dofs_list = []

for dist in model.distributions:
    means.append(dist.means)
    variances.append(dist.covs)
    dofs_list.append(dist.dofs)

print("\nState means:")
print(np.array(means))

print("\nState variances:")
print(np.array(variances))

print("\nDegrees of freedom (per state):")
print(dofs_list)

grouped = data.groupby("Regime")["Returns"]
vols = grouped.std()

volatile_state = vols.idxmax()
calm_state = vols.idxmin()

print(f"\nVolatile state: {volatile_state}")
print(f"Calm state: {calm_state}")

plt.figure(figsize=(12,6))

plt.plot(data.index, data["Close"], label="S&P 500", linewidth=1)

plt.scatter(
    data.index[data["Regime"] == volatile_state],
    data["Close"][data["Regime"] == volatile_state],
    color="red",
    s=3,
    label="Volatile State"
)

plt.scatter(
    data.index[data["Regime"] == calm_state],
    data["Close"][data["Regime"] == calm_state],
    color="green",
    s=3,
    label="Calm State"
)

plt.title("Student-t HMM Market Regimes")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()

plt.show()
