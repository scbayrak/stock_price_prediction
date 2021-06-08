import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as data

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from dataset import StockData
from earlyStop import EarlyStop
from models import StockPredictionModels
from customLoss import CustomMSELoss

device = torch.device("cpu")

if torch.cuda.is_available():
    device = torch.device("cuda")

print(f"Using: {device}")


class Training():
    """
    This class is a utility class that isresponsible to run multiple
    training with different hyperparameters. Then it stores the results and charts as *.png files
    """
    def __init__(self, rnn_type):
        self.rnn_type = rnn_type
        self.model = None
        self.epoch = None
        self.symbol = None
        self.start_date = None
        self.end_date = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.scaler = None
        self.train_loss_history = []
        self.val_loss_history = []


    def create_data_loaders(self, symbol, start_date, end_date, seq_len, batch_size):
        """
        This method creates data loader and filter the dataset based on start and end date.

        Args:
            symbol (str): Stock identified in NASDAQ stock exchange.
            start_date (date): Start date of stock price 
            end_date (date): End date pf stock price
            seq_len (int): Lenght of a sqeuence , number of days for for a given dates sequence of price
            batch_size (int): number of sqeuence in each batch
        """
        # Save the parameters to use in other functions
        self.start_date = start_date
        self.end_date = end_date
        self.symbol = symbol

        # Dataloaders
        train_data = StockData(seq_len, "train", symbol=symbol, start_date = start_date, end_date= end_date)
        self.train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=False)
        val_data = StockData(seq_len, "val", symbol=symbol, start_date = start_date, end_date= end_date)
        self.val_loader = data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
        test_data = StockData(seq_len, "test", symbol=symbol, start_date = start_date, end_date= end_date)
        self.test_loader = data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

        # We will use this scaler to inverse scale of model outputs.
        self.scaler = train_data.scaler

    def train(self, input_dim, hidden_dim, n_layers, output_dim, loss_funct, optimiser, learning_rate, epochs, test_no, save_weights=False):
        """
        This method is to train the models , it choses optimzer and iterate the training process epoch times. 
        If early stopping criteria is met , then training is stopped.

        Args:
            input_dim (int): The number of expected features in the input `x`
            hidden_dim (int): he number of features in the hidden state `h`
            n_layers (int): umber of recurrent layers.
            output_dim (int): tensor containing the output features `(h_t)` from the last layer of the LSTM
            optimiser (string): optimizer
            learning_rate (float): Learning rate
            epochs (int): number of iteration
            test_no (str): experiment identifier
        """
        # Model, Loss function
        model = StockPredictionModels(input_dim, hidden_dim, n_layers, output_dim, model=self.rnn_type)
        self.model = model.to(device)
        if loss_funct == "MSE":
            loss_funct = nn.MSELoss()
        elif loss_funct == "custom":
            loss_funct = CustomMSELoss()

        # Choose the optimizer
        if optimiser == "adam":
            optimiser = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimiser == "SGD":
            optimiser = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

        # Initiate early stop based on validation loss
        early_stopping = EarlyStop(self.model, self.rnn_type, self.symbol, test_no,save=save_weights)

        # Training loop
        for epoch in range(epochs):
            self.epoch = epoch + 1
            epoch_train_loss = 0
            epoch_val_loss = 0
            for idx, X_train, y_train in self.train_loader:
                X_train, y_train = X_train.to(device), y_train.to(device)
                output = self.model(X_train)
                train_loss = loss_funct(y_train, output)
                optimiser.zero_grad()
                train_loss = train_loss.float()
                train_loss.backward()
                optimiser.step()
                epoch_train_loss += train_loss.item()
        
            for idx, X_val, y_val in self.val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                output = self.model(X_val)
                val_loss = loss_funct(y_val, output)
                epoch_val_loss += val_loss.item()

            self.train_loss_history.append(epoch_train_loss)
            self.val_loss_history.append(epoch_val_loss)
            print(f"Epoch: {self.epoch}, Training MSE Loss:{epoch_train_loss},  Validation MSE Loss:{epoch_val_loss}")

            #implement the early stop
            early_stopping(epoch_val_loss, self.epoch)
            if early_stopping.early_stop:
                break

    def evaluate(self):
        """
        This method is used to evaluate and test the trained model.

        Returns:
            [type]: predictions, truths, test score
        """
        y_pred = []
        y_test = []
        for batch, values in enumerate(self.test_loader):
            idx, X_test, y = values
            X_test, y = X_test.to(device), y.to(device)
            self.model.eval()
            output = self.model(X_test)
            y_pred.append(output.detach().cpu().numpy()[:,0])
            y_test.append(y.detach().cpu().numpy()[:,0])

        y_test = np.concatenate(y_test)
        y_pred = np.concatenate(y_pred)

        y_pred = self.scaler.inverse_transform(y_pred)
        y_test = self.scaler.inverse_transform(y_test)

        #
        diff_pred = np.diff(y_pred)
        diff_test = np.diff(y_test)

        # increases = 2, decrease = 1
        diff_pred[diff_pred > 0] = 2
        diff_pred[diff_pred <= 0] = 1

        diff_test[diff_test > 0] = 2
        diff_test[diff_test <= 0] = 1

        # if both true and pred are increase or decrease, the difference would be 0. 
        # if result is -1 then true stock increases but prediction decreases
        # if result is +1 then true stock decreases but prediction increases
        # 0 , correct direction 
        result = diff_pred - diff_test
        # Movement Direction Accuracy
        self.testScoreMDA = np.count_nonzero(result == 0) / result.shape[0]

        # MPA - Mean Prediction Accuracy
        self.testScoreMPA = 1 - np.sum(np.abs(y_test - y_pred) / y_test) / len(y_test)

        # RMSE - Root Mean Squared Error
        self.testScoreRMSE = np.sqrt(mean_squared_error(y_test, y_pred)).astype(float)
        print(f"Test score --> RMSE: {self.testScoreRMSE}, MPA {self.testScoreMPA}, MDA : {self.testScoreMDA}")

        return y_pred, y_test, self.testScoreRMSE, self.testScoreMPA, self.testScoreMDA

    def generate_results(self, y_pred, y_test, test_no, test_dict):
        """
        This method visualize the results.
        It generates LOSS graph, prediction and truth values and a table of hyperparameters

        Args:
            y_pred ([tensor]): Predicted values
            y_test ([tensor]): True values
            testScore ([np.array]): test score
            test_no ([str]): tes Identifier
            test_dict ([dic]): test object contains all hyperparameters of experiment.
        """

        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(5, 4, wspace=0.3, hspace=2)
        fig = plt.figure(figsize=(20,5))
        fig.suptitle('Prediction Model: {}'.format(self.rnn_type))

        ax1 = plt.subplot(gs[0:4, :1], )
        ax1.set_title('Loss for: {} , RMSE test score {:.2f} , MPA test score {:.2f} % , MDA test score {:.2f} % '.format(self.rnn_type, self.testScoreRMSE , self.testScoreMPA * 100, self.testScoreMDA * 100))
        ax1.plot(range(1, self.epoch+1), self.train_loss_history, label = 'Training Loss')
        ax1.plot(range(1, self.epoch+1), self.val_loss_history, label = "Validation Loss")
        ax1.legend(loc=1)

        ax2 = plt.subplot(gs[0:4, 1:],)
        dates = pd.date_range(self.start_date, self.end_date, freq="B")
        ax2.plot(dates[-(y_test.size):], y_test, color = 'blue', label = 'Real Stock Price')
        ax2.plot(dates[-(y_test.size):], y_pred, color = 'green', label = 'Predicted Stock Price')
        ax2.set_title('{} Stock Price Prediction'.format(self.symbol))
        ax2.set(xlabel='Time', ylabel='Stock Price')
        ax2.tick_params(axis='x', rotation=30)
        ax2.legend(loc=2)

        cells = list(test_dict.values())
        cells = [str(i) for i in cells]
        columns = list(test_dict.keys())
        ax3 = plt.subplot(gs[4:, :4])
        ax3.axis('off')
        ax3.table(cellText=[cells], colLabels=columns, loc='bottom', cellLoc='center')

        plt.savefig('results/charts/Test_{}.png'.format(test_no), bbox_inches='tight')


