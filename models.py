import torch.nn as nn
import torch.nn.functional as F

# Модель із одним прихованим шарам


class ModelA(nn.Module):
    def __init__(self):
        super(ModelA, self).__init__()
        self.input_layer = nn.Linear(28*28, 128)
        self.hidden_layer = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer(x))
        x = F.log_softmax(self.output_layer(x), dim=1)
        return x

# Модель із двома прихованими шарами


class ModelB(nn.Module):
    def __init__(self):
        super(ModelB, self).__init__()
        self.input_layer = nn.Linear(28*28, 256)
        self.hidden_layer1 = nn.Linear(256, 128)
        self.hidden_layer2 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer1(x))
        x = F.relu(self.hidden_layer2(x))
        x = F.log_softmax(self.output_layer(x), dim=1)
        return x

# Модель із трьома прихованими шарами


class ModelC(nn.Module):
    def __init__(self):
        super(ModelC, self).__init__()
        self.input_layer = nn.Linear(28*28, 512)
        self.hidden_layer1 = nn.Linear(512, 256)
        self.hidden_layer2 = nn.Linear(256, 128)
        self.hidden_layer3 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer1(x))
        x = F.relu(self.hidden_layer2(x))
        x = F.relu(self.hidden_layer3(x))
        x = F.log_softmax(self.output_layer(x), dim=1)
        return x
