import torch
from torch import nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 8, 7, 2, 3),
            nn.LayerNorm(normalized_shape=[8, 32, 32]),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
        )  # in (3, 64, 64) out (8, 16, 16)
        # layer2
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 8, 3, 1, 1),
            nn.LayerNorm(normalized_shape=[8, 16, 16]),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
        )  # in (8, 16, 16) out (8, 8, 8)
        self.layer3 = nn.Sequential(
            nn.Conv2d(8, 4, 3, 1, 1),
            nn.LayerNorm(normalized_shape=[4, 8, 8]),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
        )  # in (8, 8, 8) out (4, 4, 4)
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(64, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 10),
        )  # in (4, 4, 4) out (4, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.head(x)
        return x


if __name__ == "__main__":
    model = Model()
    print(model)
    num_param = 0
    for param in model.parameters():
        num_param += param.numel()
    print("num_param:", num_param)
