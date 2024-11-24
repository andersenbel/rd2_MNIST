import torch.nn as nn
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.input_layer = nn.Linear(28*28, 128)
        self.hidden_layer = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Розгортання зображення в 1D-вектор
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer(x))
        x = F.log_softmax(self.output_layer(x), dim=1)
        return x
