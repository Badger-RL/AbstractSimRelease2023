import torch.nn as nn
import torch.nn.functional as functional

class Network(nn.Module):
    def __init__(self, history_length):
        super().__init__()
        self.dense1 = nn.Linear(10 * (history_length + 1) + 3 * history_length, 100)
        self.dense2 = nn.Linear(100, 100)
        self.dense3 = nn.Linear(100, 50)
        self.dense4 = nn.Linear(50, 50)
        self.dense5 = nn.Linear(50, 50)
        self.batch_norm1 = nn.BatchNorm1d(50)
        self.dense6 = nn.Linear(50, 50)
        self.dense7 = nn.Linear(50, 50)
        self.dense8 = nn.Linear(50, 50)
        self.dense9 = nn.Linear(50, 50)
        self.dense10 = nn.Linear(50, 50)
        self.batch_norm2 = nn.BatchNorm1d(50)
        self.dense11 = nn.Linear(50, 50)
        self.dense12 = nn.Linear(50, 50)
        self.dense13 = nn.Linear(50, 50)
        self.dense14 = nn.Linear(50, 50)
        self.dense15 = nn.Linear(50, 50)
        self.batch_norm3 = nn.BatchNorm1d(50)
        self.dense16 = nn.Linear(50, 50)
        self.dense17 = nn.Linear(50, 50)
        self.dense18 = nn.Linear(50, 50)
        self.dense19 = nn.Linear(50, 50)
        self.dense20 = nn.Linear(50, 50)
        self.batch_norm4 = nn.BatchNorm1d(50)
        self.dense21 = nn.Linear(50, 50)
        self.dense22 = nn.Linear(50, 50)
        self.dense23 = nn.Linear(50, 50)
        self.dense24 = nn.Linear(50, 50)
        self.dense25 = nn.Linear(50, 50)
        self.batch_norm5 = nn.BatchNorm1d(50)
        self.dense26 = nn.Linear(50, 50)
        self.dense27 = nn.Linear(50, 50)
        self.dense28 = nn.Linear(50, 50)
        self.dense29 = nn.Linear(50, 50)
        self.dense30 = nn.Linear(50, 50)
        self.batch_norm6 = nn.BatchNorm1d(50)
        self.dense31 = nn.Linear(50, 50)
        self.dense32 = nn.Linear(50, 50)
        self.dense33 = nn.Linear(50, 50)
        self.dense34 = nn.Linear(50, 50)
        self.dense35 = nn.Linear(50, 10)

    def forward(self, x):
        x = functional.relu(self.dense1(x))
        x = functional.relu(self.dense2(x))
        x = functional.relu(self.dense3(x))
        x = functional.relu(self.dense4(x))
        x = functional.relu(self.dense5(x))
        x = self.batch_norm1(x)
        x = functional.relu(self.dense6(x))
        x = functional.relu(self.dense7(x))
        x = functional.relu(self.dense8(x))
        x = functional.relu(self.dense9(x))
        x = functional.relu(self.dense10(x))
        x = self.batch_norm2(x)
        x = functional.relu(self.dense11(x))
        x = functional.relu(self.dense12(x))
        x = functional.relu(self.dense13(x))
        x = functional.relu(self.dense14(x))
        x = functional.relu(self.dense15(x))
        x = self.batch_norm3(x)
        x = functional.relu(self.dense16(x))
        x = functional.relu(self.dense17(x))
        x = functional.relu(self.dense18(x))
        x = functional.relu(self.dense19(x))
        x = functional.relu(self.dense20(x))
        x = self.batch_norm4(x)
        x = functional.relu(self.dense21(x))
        x = functional.relu(self.dense22(x))
        x = functional.relu(self.dense23(x))
        x = functional.relu(self.dense24(x))
        x = functional.relu(self.dense25(x))
        x = self.batch_norm5(x)
        x = functional.relu(self.dense26(x))
        x = functional.relu(self.dense27(x))
        x = functional.relu(self.dense28(x))
        x = functional.relu(self.dense29(x))
        x = functional.relu(self.dense30(x))
        x = self.batch_norm6(x)
        x = functional.relu(self.dense31(x))
        x = functional.relu(self.dense32(x))
        x = functional.relu(self.dense33(x))
        x = functional.relu(self.dense34(x))
        x = functional.relu(self.dense35(x))
        return x