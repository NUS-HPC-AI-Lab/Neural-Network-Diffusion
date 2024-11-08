import torch
import pickle
import numpy as np


with open("plot_numberckpt_001.cache", "rb") as f:
    cache001 = pickle.load(f)
with open("plot_numberckpt_010.cache", "rb") as f:
    cache010 = pickle.load(f)
with open("plot_numberckpt_050.cache", "rb") as f:
    cache050 = pickle.load(f)
with open("plot_numberckpt_200.cache", "rb") as f:
    cache200 = pickle.load(f)
with open("plot_numberckpt_500.cache", "rb") as f:
    cache500 = pickle.load(f)
caches = [cache001[0], cache010[0], cache050[0], cache200[0], cache500[0]]


for cache in caches:
    num_checkpoint = cache["num_checkpoint"]
    num_generated = cache["num_generated"]
    total_acc_list = cache["total_acc_list"]
    iou_matrix = cache["iou_matrix"]

    # sim = iou_matrix[num_checkpoint:num_checkpoint+num_generated, num_checkpoint:num_checkpoint+num_generated]
    # print((sim.sum() - num_generated) / (num_generated * num_generated - num_generated))
    # sim = iou_matrix[:num_checkpoint, num_checkpoint:num_checkpoint+num_generated]
    # print(sim.mean())
    # acc = np.array(total_acc_list[num_checkpoint:num_generated+num_checkpoint])
    # print(acc)

    


