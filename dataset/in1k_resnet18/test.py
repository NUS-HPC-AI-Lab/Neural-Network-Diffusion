import os
import sys
import torch

try:
    from finetune import *
except ImportError:
    from .finetune import *

try:
    test_item = sys.argv[1]
except IndexError:
    assert __name__ == "__main__"
    test_item = "./checkpoint"


test_items = []
if os.path.isdir(test_item):
    for item in os.listdir(test_item):
        if item.endswith('.pth'):
            item = os.path.join(test_item, item)
            test_items.append(item)
elif os.path.isfile(test_item):
    test_items.append(test_item)


if __name__ == "__main__":
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader = get_data_loaders(config)

    for item in test_items:
        print(f"Test model: {os.path.basename(item)}")
        state = torch.load(item, map_location=device, weights_only=True)
        model.load_state_dict({k: v.to(torch.float32).to(device) for k, v in state.items()}, strict=False)
        loss, acc, all_targets, all_predicts = test(model, test_loader, device)
        print(f"Loss = {loss:.4f}, Acc = {acc:.4f}\n")
