import os
import json
import random
import numpy as np
from tqdm.auto import tqdm
import timm
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_config():
    config_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
    with open(config_file, "r") as f:
        additional_config = json.load(f)
    config = {
        "batch_size": 128,
        "num_workers": 16,
        "learning_rate": 0.005,
        "weight_decay": 5e-3,
        "epochs": 1,
        "save_learning_rate": 0.005,
        "total_save_number": 200,
        "tag": os.path.basename(os.path.dirname(__file__)),
        "freeze_epochs": 0,
        "seed": 40,
        "model_name": "resnet50"
    }
    config.update(additional_config)
    return config


def get_data_loaders(config):
    print(f"Initializing data loaders with dataset_root: {config['imagenet_root']}")
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_path = os.path.join(config['imagenet_root'], 'train')
    val_path = os.path.join(config['imagenet_root'], 'val')
    # print(f"Checking if train path exists: {os.path.exists(train_path)}")
    # print(f"Checking if val path exists: {os.path.exists(val_path)}")
    train_dataset = ImageFolder(root=train_path, transform=train_transform)
    test_dataset = ImageFolder(root=val_path, transform=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,
                              num_workers=config["num_workers"], pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False,
                             num_workers=config["num_workers"], pin_memory=True)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    return train_loader, test_loader


def get_optimizer_and_scheduler(model, config):
    trainable_params = []
    last_two_norms = ['layer4.2.bn2.weight', 'layer4.2.bn2.bias', 'layer4.2.bn3.weight', 'layer4.2.bn3.bias']
    for name, param in model.named_parameters():
        if any(norm in name for norm in last_two_norms):
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False
    optimizer = optim.AdamW(trainable_params, lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(get_data_loaders(config)[0]), eta_min=config["save_learning_rate"])
    return optimizer, scheduler


@torch.no_grad()
def test(model, test_loader, device):
    model = model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    total = 0
    all_targets = []
    all_predicts = []
    pbar = tqdm(test_loader, desc='Testing', leave=False, ncols=100)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        all_targets.extend(targets.cpu().tolist())
        test_loss += loss.item()
        _, predicts = outputs.max(1)
        all_predicts.extend(predicts.cpu().tolist())
        total += targets.size(0)
        correct += predicts.eq(targets).sum().item()
        pbar.set_postfix({'Loss': f'{test_loss / (pbar.n + 1):.3f}', 'Acc': f'{100. * correct / total:.2f}%'})
    loss = test_loss / len(test_loader)
    acc = correct / total
    print(f"Test Loss: {loss:.4f} | Test Acc: {acc:.4f}")
    return loss, acc, all_targets, all_predicts


def save_checkpoint(model, batch_idx, acc, config):
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    last_two_norms = ['layer4.2.bn2.weight', 'layer4.2.bn2.bias', 'layer4.2.bn3.weight', 'layer4.2.bn3.bias']
    save_state = {key: value.cpu().to(torch.float32) for key, value in model.state_dict().items()
                  if any(norm in key for norm in last_two_norms)}
    save_path = f"checkpoint/{str(batch_idx).zfill(4)}_acc{acc:.4f}_seed{config['seed']:04d}_{config['tag']}.pth"
    torch.save(save_state, save_path)
    print(f"Saved: {save_path}")


config = get_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(config['seed'])

# load dataset
train_loader, test_loader = get_data_loaders(config)
# load model
model = timm.create_model('resnet50', pretrained=True, num_classes=1000)
model = model.to(device)
# get optimizer
optimizer, scheduler = get_optimizer_and_scheduler(model, config)


if __name__ == "__main__":
    print("Initial test:")
    test(model, test_loader, device)
    # Calculate the interval for saving checkpoints
    total_batches = len(train_loader)
    save_interval = max(1, total_batches // config["total_save_number"])
    model.train()
    criterion = nn.CrossEntropyLoss()
    pbar = tqdm(train_loader, desc='Training', ncols=100)
    # pre epoch
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        pbar.set_postfix({'Loss': f'{loss:.3f}'})
    # save epoch
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        # Save checkpoint at regular intervals
        if ((batch_idx + 1) % save_interval == 0 or batch_idx == total_batches - 1) and batch_idx > 500:
            if batch_idx % 1000 == 0:
                loss, acc, _, _ = test(model, test_loader, device)
            loss, acc = 1., 1.
            save_checkpoint(model, batch_idx, acc, config)
        pbar.set_postfix({'Loss': f'{loss:.3f}'})
    print("Fine-tuning completed.")
