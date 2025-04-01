import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

symbol = "DAX"
seq_len = 30
future_days = 30
hidden_size = 80
epochs = 256
lr = 0.0005
seed = 314

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

df = yf.download(symbol, period="2y", interval="1d")
df = df[["Close"]].dropna()

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[["Close"]])

X, y = [], []
for i in range(len(scaled) - seq_len - future_days):
    X.append(scaled[i:i + seq_len])
    y.append(scaled[i + seq_len:i + seq_len + future_days])
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=4):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, future_days)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


model = LSTM(hidden_size=hidden_size)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    out = model(X)
    loss = loss_fn(out, y.squeeze(-1))
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    last_seq = torch.tensor(scaled[-seq_len:], dtype=torch.float32).unsqueeze(0)
    pred_scaled = model(last_seq).squeeze().numpy()
    pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

dates_all = df.index
values_all = df["Close"].values
future_dates = pd.date_range(dates_all[-1] + pd.Timedelta(days=1), periods=future_days, freq='B')

with torch.no_grad():
    last_true_seq = X[-1].unsqueeze(0)
    last_pred_scaled = model(last_true_seq).squeeze().numpy()
    last_pred = scaler.inverse_transform(last_pred_scaled.reshape(-1, 1)).flatten()

true_last_month = df["Close"].values[-future_days:]
true_last_dates = dates_all[-future_days:]

mse = mean_squared_error(true_last_month, last_pred)
print(f"MSE on last {future_days} days: {mse:.4f}")

plt.figure(figsize=(14, 6))
plt.plot(dates_all, values_all, label="Real")
plt.plot(future_dates, pred, label="Forecast", color="orange")
plt.plot(true_last_dates, last_pred, label="Prediction", color="green", linestyle="--")
plt.xlabel("Date")
plt.ylabel("DAX")
plt.legend()
plt.tight_layout()
plt.show()
