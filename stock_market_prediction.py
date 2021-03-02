import os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.utils.data as data
from stock_dataset import stock_data
from LSTM import LSTM

device = torch.device("cpu")

if torch.cuda.is_available():
    device = torch.device("cuda")

print(f"Using: {device}")


start_date = '2010-09-18'
end_date = '2016-12-31'
symbol = "AAPL"
dates = pd.date_range(start_date, end_date, freq="B")

# create stock data array

def create_stock_values(symbol, dates):
    df = pd.DataFrame(index=dates)
    new_df = pd.read_csv("stocks/{}.csv".format(symbol), index_col="Date", parse_dates=True, usecols=["Date", "Close"], na_values=["nan"])
    new_df = new_df.rename(columns={"Close":symbol})
    df = df.join(new_df)
    # for symbol in symbols:
    #     current_df = pd.read_csv(os.path.join(self.dir, "Stocks/{}.us.txt".format(symbol)), index_col='Date, parse_dates=True, usecols=["Date", "Close"], na_values=[nan])
    #     current_df = current_df.rename(columns={"Date":symbol})
    #     self.df = self.df.join(current_df)
    df = df.fillna(method='ffill')
    global scaler
    scaler = StandardScaler()
    df[symbol] = scaler.fit_transform(df[symbol].values.reshape(-1,1))
    stock_values = df.values
    return stock_values

stock_values = create_stock_values(symbol, dates)

# create dataloaders
batch_size = 32

train_data = stock_data(stock_values, 60, "train")
train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=False)
val_data = stock_data(stock_values, 60, "val")
val_loader = data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_data = stock_data(stock_values, 60, "test")
test_loader = data.DataLoader(test_data, batch_size=batch_size, shuffle=False)


# Build LSTM model
input_dim = 1
hidden_dim = 32
n_layers = 2 
output_dim = 1

model = LSTM(input_dim, hidden_dim, n_layers, output_dim)
loss_funct = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

# Training

train_loss_history = []
val_loss_history = []

total_epochs = 50

for epoch in range(total_epochs):
    epoch_train_loss = 0
    epoch_val_loss = 0
    for idx, X_train, y_train in train_loader:
        X_train, y_train = X_train.to(device), y_train.to(device)
        output = model(X_train)
        train_loss = loss_funct(y_train, output)
        optimiser.zero_grad()
        train_loss = train_loss.float()
        train_loss.backward()
        optimiser.step()
        epoch_train_loss += train_loss.item()
    print(f"Epoch: {epoch} Training MSE Loss:{epoch_train_loss}")
    train_loss_history.append(epoch_train_loss)
    for idx, X_val, y_val in val_loader:
        X_val, y_val = X_val.to(device), y_val.to(device)
        output = model(X_val)
        val_loss = loss_funct(y_val, output)
        epoch_val_loss += val_loss.item()
    val_loss_history.append(epoch_val_loss)
    print(f"Epoch: {epoch} Validation MSE Loss:{epoch_val_loss}")

# plt.plot(range(total_epochs), train_loss_history)
# plt.plot(range(total_epochs), val_loss_history)
# plt.show()

# Evaluation

y_pred = []
y_test = []
for batch, values in enumerate(test_loader):
    idx, X_test, y = values
    X_test, y = X_test.to(device), y.to(device)
    model.eval()
    output = model(X_test)
    # y_pred[batch] = output.detach().numpy()[:,0]
    # y_test[batch] = y.detach().numpy()[:,0]
    y_pred.append(output.detach().numpy()[:,0])
    y_test.append(y.detach().numpy()[:,0])

y_test= np.concatenate(y_test)
y_pred = np.concatenate(y_pred)

y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)

testScore = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Test score is {testScore}")

plt.plot(dates[-(y_test.size):], y_test, color = 'blue', label = 'Real Stock Price')
plt.plot(dates[-(y_test.size):], y_pred, color = 'green', label = 'Predicted Stock Price')
plt.title('{} Stock Price Prediction'.format(symbol))
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
