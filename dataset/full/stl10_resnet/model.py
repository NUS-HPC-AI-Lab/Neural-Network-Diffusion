import torch
from torch import nn
from torch.nn import functional as F


class ResBlock(nn.Module):
    def __init__(self, channels, middle_channels):
        super().__init__()
        self.linear1 = nn.Conv2d(channels, middle_channels, 3, 1, 1)
        self.linear2 = nn.Conv2d(middle_channels, channels, 3, 1, 1)

    def forward(self, x):
        return F.leaky_relu(self.linear2(F.leaky_relu(self.linear1(x)))) + x


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Sequential(
            nn.Conv2d(3, 8, 3, 2, 1),
            nn.LeakyReLU(),
            nn.LayerNorm(normalized_shape=[8, 32, 32]),
            nn.MaxPool2d(2, 2),
        )  # out (8, 16, 16)
        # layer2
        self.layer1 = nn.Sequential(
            ResBlock(8, 4),
            nn.LayerNorm(normalized_shape=[8, 16, 16]),
            nn.MaxPool2d(2, 2),
        )  # out (8, 8, 8)
        # layer3
        self.layer2 = nn.Sequential(
            ResBlock(8, 4),
            nn.LayerNorm(normalized_shape=[8, 8, 8]),
            nn.MaxPool2d(2, 2),
        )  # out (8, 4, 4)
        # head
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(128, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 10),
        )  # in (8, 4, 4) out (4, 10)

    def forward(self, x):
        x = self.input(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.head(x)
        return x


if __name__ == "__main__":
    model = Model()
    print(model)
    num_param = 0
    for param in model.parameters():
        num_param += param.numel()
    print("num_param:", num_param)
