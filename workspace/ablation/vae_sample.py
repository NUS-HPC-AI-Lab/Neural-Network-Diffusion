import sys, os
root = os.sep + os.sep.join(__file__.split(os.sep)[1:__file__.split(os.sep).index("Neural-Network-Parameter-Diffusion")+1])
sys.path.append(root)
os.chdir(root)
USE_WANDB = False

# set global seed
import random
import numpy as np
import torch
seed = SEED = 430
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(seed)
random.seed(seed)

# other
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
if USE_WANDB: import wandb
import torch.optim as optim
# model
from model.pdiff import PDiff as Model
from model.pdiff import OneDimVAE as VAE
from model.diffusion import DDPMSampler, DDIMSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from accelerate.utils import DistributedDataParallelKwargs, AutocastKwargs
from accelerate import Accelerator
# dataset
from dataset import VAE_Sample as Dataset
from torch.utils.data import DataLoader

config = {
    "seed": SEED,
    # dataset setting
    "dataset": Dataset,
    "sequence_length": 'auto',
    # train setting
    "batch_size": 50,
    "num_workers": 4,
    "vae_steps": 2000,  # vae training steps
    "vae_learning_rate": 0.0002,  # vae learning rate
    "weight_decay": 0.0,
    "save_every": 500,
    "print_every": 50,
    "kld_weight": 1.0,
    "autocast": lambda i: True,
    "checkpoint_save_path": "./checkpoint",
    # test setting
    "test_batch_size": 1,  # fixed, don't change this
    "generated_path": Dataset.generated_path,
    "test_command": Dataset.test_command,
    # to log
    "model_config": {
        # diffusion config
        "layer_channels": [1, 64, 128, 256, 512, 256, 128, 64, 1],  # channels of 1D CNN
        "model_dim": 128,  # latent dim of vae
        "kernel_size": 7,
        "sample_mode": DDPMSampler,
        "beta": (0.0001, 0.02),
        "T": 1000,
        # vae config
        "channels": [64, 128, 256, 256, 32],
    },
    "tag": os.path.basename(__file__)[:-3],
}

# Data
divide_slice_length = 64
print('==> Preparing data..')
# import pdb; pdb.set_trace()
train_set = config["dataset"](dim_per_token=divide_slice_length,
                              granularity=0,
                              pe_granularity=0,
                              fill_value=0.)
print("Dataset length:", train_set.real_length)
print("input shape:", train_set[0][0].flatten().shape)
if config["sequence_length"] == "auto":
    config["sequence_length"] = train_set.sequence_length * divide_slice_length
    print(f"sequence length: {config['sequence_length']}")
train_loader = DataLoader(
    dataset=train_set,
    batch_size=config["batch_size"],
    num_workers=config["num_workers"],
    persistent_workers=True,
    drop_last=True,
    shuffle=True,
)

# Model
print('==> Building model..')
vae = VAE(d_model=config["model_config"]["channels"],
          d_latent=config["model_config"]["model_dim"],
          sequence_length=config["sequence_length"],
          kernel_size=config["model_config"]["kernel_size"])

# Optimizer
print('==> Building optimizer..')
optimizer = optim.AdamW(
    params=vae.parameters(),
    lr=config["vae_learning_rate"],
    weight_decay=config["weight_decay"],
)
scheduler = CosineAnnealingLR(
    optimizer=optimizer,
    T_max=config["vae_steps"],
)

# accelerator
if __name__ == "__main__":
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs,])
    vae, optimizer, train_loader = \
            accelerator.prepare(vae, optimizer, train_loader)

# wandb
if __name__ == "__main__" and USE_WANDB and accelerator.is_main_process:
    wandb.login(key="your_api_key")
    wandb.init(project="AR-Param-Generation", name=config['tag'], config=config,)




# Training
print('==> Defining training..')

def train_vae():
    if not USE_WANDB:
        train_loss = 0
        this_steps = 0
    print("==> start training vae..")
    vae.train()
    for batch_idx, (param, _) in enumerate(train_loader):
        optimizer.zero_grad()
        with accelerator.autocast(autocast_handler=AutocastKwargs(enabled=config["autocast"](batch_idx))):
            param = param.flatten(start_dim=1)
            # param += torch.randn_like(param) * 0.001
            loss = vae(x=param, use_var=True, manual_std=None, kld_weight=config["kld_weight"])
        accelerator.backward(loss)
        optimizer.step()
        if accelerator.is_main_process:
            scheduler.step()
        if USE_WANDB and accelerator.is_main_process:
            wandb.log({"vae_loss": loss.item()})
        elif USE_WANDB:
            pass
        else:
            train_loss += loss.item()
            this_steps += 1
            if this_steps % config["print_every"] == 0:
                print('Loss: %.6f' % (train_loss/this_steps))
                this_steps = 0
                train_loss = 0
        if batch_idx >= config["vae_steps"]:
            break

def generate(save_path=config["generated_path"], need_test=True):
    print("\n==> Generating..")
    vae.eval()
    with torch.no_grad():
        prediction = vae.sample()
        generated_norm = prediction.abs().mean()
    print("Generated_norm:", generated_norm.item())
    if USE_WANDB:
        wandb.log({"generated_norm": generated_norm.item()})
    prediction = prediction.view(-1, divide_slice_length)
    train_set.save_params(prediction, save_path=save_path)
    if need_test:
        os.system(config["test_command"])
        print("\n")
    vae.train()
    return prediction

if __name__ == '__main__':
    train_vae()
    vae = accelerator.unwrap_model(vae)
    generate()




# generate
print('==> Defining generate..')
def generate(save_path=config["generated_path"], test_command=config["test_command"], need_test=True):
    print("\n==> Generating..")
    vae.eval()
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        with torch.no_grad():
            prediction = vae.sample()
            generated_norm = prediction.abs().mean()
    print("Generated_norm:", generated_norm.item())
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    train_set.save_params(prediction, save_path=save_path)
    if need_test:
        os.system(test_command)
        print("\n")

generate_config = {
    "device": "cuda",
    "num_generated": 200,
    "checkpoint": f"./checkpoint/{config['tag']}.pth",
    "generated_path": os.path.join(Dataset.generated_path.rsplit("/", 1)[0], "generated_{}_{}.pth"),
    "test_command": os.path.join(Dataset.test_command.rsplit("/", 1)[0], "generated_{}_{}.pth"),
    "need_test": False,
}
config.update(generate_config)

if __name__ == "__main__":
    for i in range(config["num_generated"]):
        index = str(i+1).zfill(3)
        print("Save to", config["generated_path"].format(config["tag"], index))
        generate(
            save_path=config["generated_path"].format(config["tag"], index),
            test_command=config["test_command"].format(config["tag"], index),
            need_test=config["need_test"],
        )
