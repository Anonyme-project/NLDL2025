import torch
import torch.nn as nn
import numpy as np

def getActivationLayer(activation: str):
    if not activation: return None
    activation = activation.lower()
    if activation == "tanh": return nn.Tanh()
    elif activation == "elu": return nn.ELU()
    elif activation == "gelu": return nn.GELU()
    elif activation == "relu": return nn.ReLU()
    elif activation == "lrelu": return nn.LeakyReLU()
    elif activation == "softmax": return nn.Softmax(dim=-1)
    elif activation == "log_softmax": return nn.LogSoftmax(dim=-1)
    elif activation == "sigmoid": return nn.Sigmoid()
    elif activation == "none": return nn.Identity()

def getModel(use_mode, model_params):
    use_lstm = use_mode["lstm"]
    if use_lstm is not None and use_lstm:
        model = PINN_LSTM(**model_params)
    else:
        model = PINN_MLP(**model_params)

    return model

# =============================================================================
# Define the PINN-SIR Neural Network Model
# =============================================================================
# 1. MLP
class PINN_MLP(nn.Module):
    def __init__(self, in_dim, out_dim, h_dim, n_layers, activation="tanh"):
        super(PINN_MLP, self).__init__()
        self.in_layer = nn.Linear(in_dim, h_dim)
        h_dims = [h_dim] + [h_dim] * n_layers
        layer_list = []
        for i in range(n_layers):
            layer_list.append(getActivationLayer(activation))
            layer_list.append(nn.Linear(h_dims[i], h_dims[i+1]))
        if n_layers > 0: self.net = nn.Sequential(*layer_list)
        else: self.net = None
        self.activation = getActivationLayer(activation)
        self.out_layer = nn.Linear(h_dims[-1], out_dim)

    def forward(self, x, use_cudnn=True):
        # t should be of shape (N, 1)
        h = self.in_layer(x)
        if self.net is not None: h = self.net(h)
        h = self.activation(h)
        out = self.out_layer(h)
        # Use softplus to ensure outputs are nonnegative
        I = torch.nn.functional.softplus(out[:, 0:1])
        R = torch.nn.functional.softplus(out[:, 1:2])
        return I, R

# 1. LSTM
class PINN_LSTM(nn.Module):
    def __init__(self, in_dim, out_dim, h_dim, n_layers, activation="tanh"):
        super(PINN_LSTM, self).__init__()
        self.cat_dim = 1
        self.lstm = nn.LSTM(in_dim, h_dim, batch_first=True)
        self.fc = nn.Linear(h_dim + self.cat_dim, out_dim)  # +1 for target t

    def forward(self, x, use_cudnn=True):
        with torch.backends.cudnn.flags(enabled=use_cudnn):
            x_seq, x_last = x[:, :-1, :], x[:, -1, 0:self.cat_dim]
            lstm_out, _ = self.lstm(x_seq)          # [batch, look_back, h_dim]
            lstm_last = lstm_out[:, -1, :]          # [batch, h_dim]
            combined = torch.cat([lstm_last, x_last], dim=1)  # [batch, h_dim + self.cat_dim]
            out = self.fc(combined)               # [batch, out_dim]
            I = torch.nn.functional.softplus(out[:, 0:1])
            R = torch.nn.functional.softplus(out[:, 1:2])
            return I, R
