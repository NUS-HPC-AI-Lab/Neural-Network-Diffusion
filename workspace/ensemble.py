import sys, os
root = os.sep + os.sep.join(__file__.split(os.sep)[1:__file__.split(os.sep).index("Neural-Network-Diffusion")+1])
sys.path.append(root)
os.chdir(root)
import torch


def add(ckpt1, ckpt2):
    for k, v in ckpt1.items():
        v += ckpt2[k]
    return ckpt1

def divide(ckpt, division):
    for k, v in ckpt.items():
        v /= float(division)
    return ckpt


if len(sys.argv) == 2:
    item_list = [
        os.path.abspath(os.path.join(f"./dataset/{sys.argv[1]}", item)) for item in os.listdir(f"./dataset/{sys.argv[1]}")
        if os.path.isdir(os.path.join(f"./dataset/{sys.argv[1]}", item))
    ]  # test all area
elif len(sys.argv) == 3:
    item_list = [os.path.abspath(f"./dataset/{sys.argv[1]}/{sys.argv[2]}")]
else:  # error
    raise RuntimeError(f"len(sys.argv) = {len(sys.argv)}. It should be 2 or 3.")


print(item_list)
item_list.sort()
for item in item_list:

    # collect checkpoints
    checkpoint_folder = os.path.join(item, "checkpoint")
    sum_ckpt = None
    sum_number = 0
    for ckpt in os.listdir(checkpoint_folder):
        checkpoint_path = os.path.join(checkpoint_folder, ckpt)
        diction = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
        sum_ckpt = diction if sum_ckpt is None else add(sum_ckpt, diction)
        sum_number += 1
    sum_ckpt = divide(sum_ckpt, sum_number)

    # save and test
    test_script = os.path.join(item, "test.py")
    ensemble_ckpt = os.path.join(item, "ensemble.pth")
    torch.save(sum_ckpt, ensemble_ckpt)
    try:  # device
        device_index = os.environ["CUDA_VISIBLE_DEVICES"]
    except KeyError:
        device_index = "0"
    print(f"Testing checkpoint: {ensemble_ckpt}.")
    os.system(f"CUDA_VISIBLE_DEVICES={device_index} python {test_script} {ensemble_ckpt}")
    print("...Test finished!...\n")