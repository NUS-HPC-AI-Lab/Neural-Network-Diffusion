import os

import seaborn as sns
import pickle
from matplotlib import pyplot as plt
import numpy as np

print(os.chdir("/home/wangkai/cvpr_pdiff_cleanup/Neural-Network-Parameter-Diffusion"))

with open("plot_numberckpt_001.cache", "rb") as f:
    cache001 = pickle.load(f)
with open("plot_numberckpt_010.cache", "rb") as f:
    cache010 = pickle.load(f)
with open("plot_numberckpt_050.cache", "rb") as f:
    cache050 = pickle.load(f)
with open("plot_numberckpt_200.cache", "rb") as f:
    cache200 = pickle.load(f)
with open("plot_numberckpt_300.cache", "rb") as f:
    cache500 = pickle.load(f)

caches = [cache001[-5], cache010[-5], cache050[-5], cache200[-5], cache500[-5]]



for cache, label in zip(reversed(caches), reversed(["001", "010", "050", "200", "300"])):
    sns.scatterplot(x=cache["x"], y=cache["y"], label=cache["label"]+label)

plt.savefig("./temp.png")

# # sim = iou_matrix[num_checkpoint:num_checkpoint+num_generated, num_checkpoint:num_checkpoint+num_generated]
# # print((sim.sum() - num_generacted) / (num_generated * num_generated - num_generated))
# sim = iou_matrix[:num_checkpoint, num_checkpoint:num_checkpoint+num_generated]
# print(np.max(sim, axis=1).min())
# # acc = np.array(total_acc_list[num_checkpoint:num_generated+num_checkpoint]).mean()
# # print(acc)




