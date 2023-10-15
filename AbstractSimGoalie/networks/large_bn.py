import torch.nn as nn
import torch.nn.functional as functional

class Network(nn.Module):
    def __init__(self, history_length):
        super().__init__()
        self.dense1 = nn.Linear(10 * (history_length + 1) + 3 * history_length, 300)
        self.dense2 = nn.Linear(300, 200)
        self.batchnorm1 = nn.BatchNorm1d(200)
        self.dense3 = nn.Linear(200, 100)
        self.batchnorm2 = nn.BatchNorm1d(100)
        self.dense4 = nn.Linear(100, 50)
        self.batchnorm3 = nn.BatchNorm1d(50)
        self.dense5 = nn.Linear(50, 10)

    def forward(self, x):
        x = functional.relu(self.dense1(x))
        x = functional.relu(self.dense2(x))
        x = self.batchnorm1(x)
        x = functional.relu(self.dense3(x))
        x = self.batchnorm2(x)
        x = functional.relu(self.dense4(x))
        x = self.batchnorm3(x)
        x = self.dense5(x)
        return x