import torch
import torch.nn as nn
import torch.utils.data as data

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, output_dim):
        super(LSTM, self).__init__()
        # Hidden layer dimension
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.n_layers = n_layers

        # batch_first=True causes input/output tensors to be of shape:
        # (batch_dim, seq_dim, input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)

        # Regression layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # hidden and cell dimensions are: (n_layers, batch_size, hidden_dim)
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).requires_grad_()

        # returns: output, (h_n, c_n)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        #  output shape: batch, seq_len, num_directions * hidden_size
        # -1 takes only the output from the last in the sequence
        out = out[:, -1, :]
        out = self.fc(out)

        return out