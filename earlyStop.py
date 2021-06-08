import numpy as np
import torch

class EarlyStop():
    """
    Early stop implementation for trainin process
    If the validation loss is growing then it is better to stop to prevent overfitting
    """
    def __init__(self, model, rnn_type, symbol, test_no, patience=8, save=False):
        """
        Args:
            model ([type]): torch model object for checkpointing
            rnn_type ([type]): LSTM or GRU model 
            symbol ([type]): stock symbol in NASDAQ
            test_no ([type]): test identifier
            patience (int, optional):  Defaults to 8.
            save (bool, optional):  Defaults to False.
        """
        self.model = model
        self.rnn_type = rnn_type
        self.symbol = symbol
        self.patience = patience
        self.counter = 0
        self.min_loss = np.Inf
        self.save = save
        self.test_no = test_no
        self.early_stop = False
    
    def __call__(self, val_loss, epoch):
        if val_loss > self.min_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print("Early stopping based on validation loss")
        else:
            self.min_loss = val_loss
            if self.save:
                self.save_model(epoch)
            self.counter = 0
    
    def save_model(self, epoch):
        filename = self.rnn_type+ "_" + self.symbol + "_Test_" + str(self.test_no)
        torch.save(self.model.state_dict(), "weights/" + filename+".pth")
