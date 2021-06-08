import torch
import torch.nn as nn
import torch.utils.data as data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class StockPredictionModels(nn.Module):
    """Stock Prediction model
    It uses LSTM and GRU model to predict NASDAQ stock prices. 

    Args:
        nn ([type]): [description]
    """
    def __init__(self, input_dim, hidden_dim, n_layers, output_dim, model="LSTM"):
        """
        Args:
            iinput_dim (int): The number of expected features in the input `x`
            hidden_dim (int): he number of features in the hidden state `h`
            n_layers (int): umber of recurrent layers.
            output_dim (int): tensor containing the output features `(h_t)` from the last layer of the LSTM
            model (str, optional): Defaults to "LSTM".
        """
        super(StockPredictionModels, self).__init__()

        self.model = model
        # Hidden layer dimension
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.n_layers = n_layers

        # batch_first=True causes input/output tensors to be of shape:
        # (batch_dim, seq_dim, input_dim)
        if self.model == "LSTM":
            self.prediction_model = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        else:
            self.prediction_model = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)    
        
        # Regression layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Feed forward of neural network

        Args:
            x ([tensor]): input sequence

        Returns:
            [tensor]: output of fully connected layer 
        """
        # hidden and cell dimensions are: (n_layers, batch_size, hidden_dim)
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).requires_grad_()
        
        h0, c0 = h0.to(device), c0.to(device)

        # returns: output, (h_n, c_n)
        if self.model == "LSTM":
            out, (hn, cn) = self.prediction_model(x, (h0.detach(), c0.detach()))
        else:
            out, (hn) = self.prediction_model(x, (h0.detach()))    

        #  output shape: batch, seq_len, num_directions * hidden_size
        # -1 takes only the output from the last in the sequence
        out = out[:, -1, :]
        out = self.fc(out)

        return out
