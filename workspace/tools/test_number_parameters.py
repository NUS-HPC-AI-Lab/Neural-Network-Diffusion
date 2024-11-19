import torch
import sys


diction = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
param = 0
for k, v in diction.items():
    param += v.numel()
print(param)
