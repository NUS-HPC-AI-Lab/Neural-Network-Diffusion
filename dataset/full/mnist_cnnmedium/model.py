from torch import nn


class CNNMedium(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, 3),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(),
            nn.Conv2d(32, 15, 3),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(),
            nn.Flatten(start_dim=1),
        )
        self.head = nn.Sequential(
            nn.Linear(60, 20),
            nn.LeakyReLU(),
            nn.Linear(20, 10),
        )

    def forward(self, x):
        x = x.view(x.size(0), 1, 32, 32).repeat(1, 3, 1, 1)
        x = self.module(x)
        x = self.head(x)
        return x


def Model():
    model = CNNMedium()
    return model


if __name__ == "__main__":
    model = Model()
    print(model)
    num_param = 0
    for param in model.parameters():
        num_param += param.numel()
    print("num_param:", num_param)
