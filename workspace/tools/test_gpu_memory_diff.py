import sys, os
root = os.sep + os.sep.join(__file__.split(os.sep)[1:__file__.split(os.sep).index("Neural-Network-Diffusion")+1])
sys.path.append(root)
os.chdir(root)


import torch
from model.pdiff import PDiff as Model
from model.pdiff import OneDimVAE as VAE
from model.diffusion import DDPMSampler, DDIMSampler
import torch.optim as optim


config = {
    "num_params": 2048,
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

Model.config = config["model_config"]
model = Model(sequence_length=config["num_params"]).cuda()
vae = VAE(
    d_model=config["model_config"]["channels"],
    d_latent=config["model_config"]["model_dim"],
    sequence_length=config["num_params"],
    kernel_size=config["model_config"]["kernel_size"]
).cuda()
optimizer = optim.AdamW(
    params=model.parameters(),
    lr=0.0001,
    weight_decay=0.0,
)


x = torch.randn(50, config["num_params"]).cuda()
for _ in range(10):
    optimizer.zero_grad()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            mu, _ = vae.encode(x)
        loss = model(x=mu)
        with torch.no_grad():
            y = vae.decode(mu)
    loss.backward()
    optimizer.step()
os.system("nvidia-smi")
input("Press enter to exit...")
