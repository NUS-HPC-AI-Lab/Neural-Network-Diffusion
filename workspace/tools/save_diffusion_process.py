import sys, os
root = os.sep + os.sep.join(__file__.split(os.sep)[1:__file__.split(os.sep).index("Neural-Network-Diffusion")+1])
sys.path.append(root)
os.chdir(root)

# torch
import gc
import torch
from copy import deepcopy
# father
import importlib
item = importlib.import_module(f"workspace.ablation.numberckpt_200")
Dataset = item.Dataset
train_set = item.train_set
config = item.config
model = item.model
vae = item.vae
config["tag"] = config.get("tag") if config.get("tag") is not None else os.path.basename(item.__file__)[:-3]


generate_config = {
    "device": "cuda",
    "checkpoint": f"./checkpoint/{config['tag']}.pth",
    "generated_path": os.path.join(Dataset.generated_path.rsplit("/", 2)[0], "process_{}/process_{}_{}.pth"),
    "interval": 2,
    "seed": 2024,
}
config.update(generate_config)




def set_global_seed():
    import random
    import numpy as np
    import torch
    seed = config["seed"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)
set_global_seed()


# Model
print('==> Building model..')
diction = torch.load(config["checkpoint"], map_location="cpu", weights_only=True)
vae.load_state_dict(diction["vae"])
model.load_state_dict(diction["diffusion"])
model = model.to(config["device"])
vae = vae.to(config["device"])


# generate
print('==> Defining generate..')
def generate():
    print("\n==> Generating..")
    model.eval()
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        with torch.no_grad():
            mus = model(sample=True, only_return_x_0=False, interval=config["interval"])
            mus = mus.view(-1, mus.size(-1))
            predictions = []
            for mu in mus:
                this_vae = deepcopy(vae)
                prediction = this_vae.decode(mu)
                predictions.append(prediction)
                del this_vae
                gc.collect()
    return predictions




if __name__ == "__main__":
    predictions = generate()
    save_path = config["generated_path"].format(
        config["seed"], config["tag"], "xxxx")
    print("Save to", save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    for i, prediction in enumerate(predictions):
        save_path = config["generated_path"].format(
            config["seed"], config["tag"], str(i * config["interval"]).zfill(4))
        train_set.save_params(prediction, save_path=save_path)
