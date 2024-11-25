import sys, os
root = os.sep + os.sep.join(__file__.split(os.sep)[1:__file__.split(os.sep).index("Neural-Network-Parameter-Diffusion")+1])
sys.path.append(root)
os.chdir(root)

import random
import pandas as pd
import numpy as np
import torch
import pickle
import importlib
item = importlib.import_module(f"dataset.{sys.argv[1]}.{sys.argv[2]}.finetune")
loader = item.test_loader
model = item.model
test = item.test
device = item.device
# tag in this file is only the name of specific dirname instead of config["tag"]
tag = os.path.basename(os.path.dirname(item.__file__))




config = {
    "checkpoint_path": f"./dataset/{sys.argv[1]}/{tag}/checkpoint",
    "generated_path": f"./dataset/{sys.argv[1]}/{tag}/generated",
    "cache_file": None,  # None means default dataset/tag/cache.pt
    "resume": True,  # if you updated the checkpoint and generated models, use "resume": False.
    "noise_intensity": [0.01, 0.03, 0.05, 0.10],
    "total_noised_number": 20,
}
assert config["total_noised_number"] % len(config["noise_intensity"]) == 0, \
    "total_noised_number must be a multiple of noise_intensity"
globals().update(config)
if len(sys.argv) == 4:
    exec("noise_intensity=[" + sys.argv[3] + "]")
    total_noised_number = len(noise_intensity) * 50
    print("noise_intensity:", noise_intensity)
    print("total_noised_number:", total_noised_number)




# load paths
checkpoint_items = [os.path.join(checkpoint_path, i) for i in os.listdir(checkpoint_path)]
generated_items = [os.path.join(generated_path, i) for i in os.listdir(generated_path)]
generated_items.sort()
total_items = list(checkpoint_items) + list(generated_items)
num_checkpoint = len(checkpoint_items)
num_generated = len(generated_items)


# define compute IoU
@torch.no_grad()
def compute_wrong_indices(diction):
    model.load_state_dict(diction, strict=False)
    model.eval()
    _, acc, all_targets, all_predicts = test(model=model, test_loader=loader, device=device)
    not_agreement = torch.logical_not(torch.eq(torch.tensor(all_targets), torch.tensor(all_predicts)))
    return not_agreement, acc

def compute_wrong_iou(a, b):
    inter = np.logical_and(a, b)
    union = np.logical_or(a, b)
    iou = np.sum(inter) / np.sum(union)
    return iou


# prepare evaluate
print("\n==> start evaluating..")
total_result_list = []
total_acc_list = []

cache_file = os.path.join(os.path.dirname(checkpoint_path), "performance.cache") if cache_file is None else cache_file
if resume is True and os.path.exists(cache_file):
    print(f"load cache from {cache_file}")
    with open(cache_file, "rb") as f:
        total_result_list, total_acc_list = pickle.load(f)
else:  # compute checkpoint and generated
    print(f"start inferencing on {tag}")
    for i, item in enumerate(total_items):
        print(f"start: {i+1}/{len(total_items)}")
        item = torch.load(item, map_location='cpu', weights_only=True)
        result, acc = compute_wrong_indices(item)
        result = result.numpy()
        total_result_list.append(result)
        total_acc_list.append(acc)
    with open(cache_file, "wb") as f:
        pickle.dump([total_result_list, total_acc_list], f)


# compute noised
checkpoint_items_for_noise = checkpoint_items.copy()
random.shuffle(checkpoint_items_for_noise)
num_each_noised = total_noised_number // len(noise_intensity)
num_noise_class = len(noise_intensity)
bias = 0
for this_noise_intensity in noise_intensity:
    for i in range(num_each_noised):
        i = i + bias
        print(f"testing noised: {i+1}/{num_each_noised * num_noise_class}")
        item = checkpoint_items_for_noise[i % num_checkpoint]
        item = torch.load(item, map_location="cpu", weights_only=True)
        new_diction = {}
        for key, value in item.items():
            if ("num_batches_tracked" in key) or (value.numel() == 1) or not torch.is_floating_point(value):
                pass  # not add noise to these
            elif "running_var" in key:
                pre_mean = value.mean() * 0.95
                value = torch.log(value / pre_mean + 0.05)
                mean, std = value.mean(), value.std()
                value = (value - mean) / std
                value += torch.randn_like(value) * this_noise_intensity
                value = value * std + mean
                value = torch.clip(torch.exp(value) - 0.05, min=0.001) * pre_mean
            else:  # conv & linear
                mean, std = value.mean(), value.std()
                value = (value - mean) / std
                value += torch.randn_like(value) * this_noise_intensity
                value = value * std + mean
            new_diction[key] = value
        result, acc = compute_wrong_indices(new_diction)
        result = result.numpy()
        total_result_list.append(result)
        total_acc_list.append(acc)
    bias += num_each_noised


# compute iou_metrix
print("start computing IoU...")
total_num = num_checkpoint + num_generated + num_each_noised * num_noise_class
assert total_num == len(total_result_list), \
        f"total_num:{total_num}, len(total_result_list):{len(total_result_list)}"
iou_matrix = np.zeros(shape=[total_num, total_num])
for i in range(total_num):
    for j in range(total_num):
        iou = compute_wrong_iou(total_result_list[i], total_result_list[j])
        iou_matrix[i, j] = iou


# save result
df = pd.DataFrame(iou_matrix)
df.to_excel(f"./iou_{tag}.xlsx", index=False)
print(f"finished Saving ./iou_{tag}.xlsx!")


log_file = os.path.join(os.path.dirname(cache_file), "log.txt")
log_file = open(log_file, "w")
ori_print = print
def print(*args, **kwargs):
    ori_print(*args, **kwargs)
    ori_print(*args, **kwargs, file=log_file)


# print summary
print("\n\n===============================================")
print(f"Summary: {tag}")
print()
print("num_checkpoint:", num_checkpoint)
print("num_generated:", num_generated)
print(f"num_noised: {num_each_noised}x{num_noise_class}")
print(f"original_acc_mean:", np.array(total_acc_list[:num_checkpoint]).mean())
print(f"original_acc_max:", np.array(total_acc_list[:num_checkpoint]).max())
print(f"original_acc_min:", np.array(total_acc_list[:num_checkpoint]).min())
print(f"original_acc_std:", np.array(total_acc_list[:num_checkpoint]).std())
print(f"original_acc_median:", np.median(np.array(total_acc_list[:num_checkpoint])))
print(f"generated_acc_mean:", np.array(total_acc_list[num_checkpoint:num_checkpoint+num_generated]).mean())
print(f"generated_acc_max:", np.array(total_acc_list[num_checkpoint:num_checkpoint+num_generated]).max())
print(f"generated_acc_min:", np.array(total_acc_list[num_checkpoint:num_checkpoint+num_generated]).min())
print(f"generated_acc_std:", np.array(total_acc_list[num_checkpoint:num_checkpoint+num_generated]).std())
print(f"generated_acc_median:", np.median(np.array(total_acc_list[num_checkpoint:num_checkpoint+num_generated])))


this_start = num_checkpoint + num_generated
for this_noise_intensity in noise_intensity:
    print(f"noise={this_noise_intensity:.4f}_acc_mean:",
          np.array(total_acc_list[this_start:this_start+num_each_noised]).mean())
    this_start += num_each_noised
print()  # empty line
origin_origin = iou_matrix[:num_checkpoint, :num_checkpoint]
origin_origin = (np.sum(origin_origin) - num_checkpoint) / (num_checkpoint * (num_checkpoint - 1))
print("origin-origin:", origin_origin)
generated_generated = iou_matrix[num_checkpoint:num_checkpoint + num_generated,
                                 num_checkpoint:num_checkpoint + num_generated]
generated_generated = (np.sum(generated_generated) - num_generated) / (num_generated * (num_generated - 1))
print("generated-generated:", generated_generated)
origin_generated = iou_matrix[num_checkpoint:num_checkpoint + num_generated, :num_checkpoint]
origin_generated = np.mean(origin_generated)
print("origin-generated:", origin_generated)
origin_generated_max = iou_matrix[num_checkpoint:num_checkpoint + num_generated, :num_checkpoint]
origin_generated_max = np.amax(origin_generated_max, axis=-1)
print("origin-generated(max):", origin_generated_max.mean())


# print noised
noised_max_list = []
this_start = num_checkpoint + num_generated
for this_noise_intensity in noise_intensity:
    print(f"\nnoise_intensity={this_noise_intensity}")
    noised_noised = iou_matrix[this_start:this_start + num_each_noised, this_start:this_start + num_each_noised]
    noised_noised = (np.sum(noised_noised) - num_each_noised) / (num_each_noised * (num_each_noised - 1))
    print("noised-noised:", noised_noised)
    origin_noised = iou_matrix[this_start:this_start + num_each_noised, :num_checkpoint]
    origin_noised = np.mean(origin_noised)
    print("origin-noised:", origin_noised)
    origin_noised_max = iou_matrix[this_start:this_start + num_each_noised, :num_checkpoint]
    origin_noised_max = np.amax(origin_noised_max, axis=-1)
    noised_max_list.append(origin_noised_max)
    print("origin-noised(max):", origin_noised_max.mean())
    this_start += num_each_noised


# save summary
summary = {
    "summary": tag,
    "num_checkpoint": num_checkpoint,
    "num_generated": num_generated,
    "num_each_noised": num_each_noised,
    "num_noise_class": num_noise_class,
    "noise_intensity": noise_intensity,
    "total_acc_list": total_acc_list,
    "iou_matrix": iou_matrix,
}
draw_cache = [summary,]
# final draw
print("\n==> start drawing..")
import seaborn as sns
import matplotlib.pyplot as plt
# origin
draw_origin_origin_max = np.amax(iou_matrix[:num_checkpoint, :num_checkpoint] - np.eye(num_checkpoint), axis=-1)
draw_origin_origin_acc = np.array(total_acc_list[:num_checkpoint])
sns.scatterplot(x=draw_origin_origin_max, y=draw_origin_origin_acc, label="origin")
draw_cache.append(dict(x=draw_origin_origin_max, y=draw_origin_origin_acc, label="origin"))
# generated
draw_origin_generated_max = origin_generated_max
draw_origin_generated_acc = np.array(total_acc_list[num_checkpoint:num_checkpoint + num_generated])
sns.scatterplot(x=draw_origin_generated_max, y=draw_origin_generated_acc, label="generated")
draw_cache.append(dict(x=draw_origin_generated_max, y=draw_origin_generated_acc, label="generated"))
# noised
this_start = num_checkpoint + num_generated
for i, this_noise_intensity in enumerate(noise_intensity):
    draw_origin_noised_max = noised_max_list[i]
    draw_origin_noised_acc = total_acc_list[this_start: this_start+num_each_noised]
    sns.scatterplot(x=draw_origin_noised_max, y=draw_origin_noised_acc, label=f"noise={this_noise_intensity:.4f}")
    draw_cache.append(dict(x=draw_origin_noised_max, y=draw_origin_noised_acc, label=f"noise={this_noise_intensity:.4f}"))
    this_start += num_each_noised
# draw
plt.savefig(f'plot_{tag}.png')
with open(f'plot_{tag}.cache', "wb") as f:
    pickle.dump(draw_cache, f)
print(f"plot saved to plot_{tag}.png")


log_file.close()
