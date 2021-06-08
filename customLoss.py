import torch
import torch.nn as nn

class CustomMSELoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred):

        # find the price differences between consecutive days
        true_price_diff = y_true[1:] - y_true[:-1]
        pred_price_diff = y_pred[1:] - y_pred[:-1]

        # convert to True if price change positive otherwise false
        y_true_movement = (true_price_diff>0)
        y_pred_movement = (pred_price_diff>0)

        # get the indices where the the movement is different
        false_indices = torch.where((y_true_movement != y_pred_movement))

        # add 1 to convert to match the indexes in y_pred
        false_indices = list(false_indices)
        false_indices[0] += 1

        # calculate square error and multiply by 1000 for wrong directions
        squared_error = (y_pred - y_true) ** 2
        squared_error[false_indices] = squared_error[false_indices] * 10000

        # get mean square error
        loss = torch.mean(squared_error)

        return loss