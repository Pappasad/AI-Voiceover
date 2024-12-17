import torch
import torch.nn as nn
import torch.optim as optim

class RVCNetwork(nn.Module):

    def __init__(self, input_size=80, hidden_size=256, num_layers=2):
        super(RVCNetwork, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fully_connected = nn.Linear(hidden_size, input_size)

        

    def forward(self, input):
        encoded, _ = self.encoder(input)
        decoded, _ = self.decoder(encoded)
        output = self.fully_connected(decoded)
        return output
    


