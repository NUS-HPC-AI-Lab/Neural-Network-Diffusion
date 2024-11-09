import sys, os
root = os.sep + os.sep.join(__file__.split(os.sep)[1:__file__.split(os.sep).index("Neural-Network-Parameter-Diffusion")+1])
sys.path.append(root)
os.chdir(root)

import pickle
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt




tags = ["numberckpt_001", "numberckpt_010", "numberckpt_050", "numberckpt_200", "numberckpt_300"]
caches = []
for tag in tags:
    with open(f"./results/plot_{tag}.cache", "rb") as f:
        caches.append(pickle.load(f))
multi_points = [cache[-5] for cache in caches]
similarity = np.array([points["x"] for points in multi_points]).T
accuracy = np.array([points["y"] for points in multi_points]).T


ax1 = sns.violinplot(data=similarity)
plt.xticks(ticks=range(len(tags)), labels=[str(int(tag[-3:])) for tag in tags])
ax1.set_ylim(0.81, 1.00)
ax2 = ax1.twinx()
ax2.plot(range(len(tags)), accuracy.max(axis=0), color='red', marker='o')
ax2.set_ylim(0.735, 0.778)


# 设置右侧 y 轴的标签
ax2.set_ylabel('Mean Similarity Score')


# for cache, label in zip(reversed(caches), reversed(["001", "010", "050", "200", "300"])):
#     sns.scatterplot(x=cache["x"], y=cache["y"], label=cache["label"]+label)
#
plt.savefig("./temp.png")

# # sim = iou_matrix[num_checkpoint:num_checkpoint+num_generated, num_checkpoint:num_checkpoint+num_generated]
# # print((sim.sum() - num_generacted) / (num_generated * num_generated - num_generated))
# sim = iou_matrix[:num_checkpoint, num_checkpoint:num_checkpoint+num_generated]
# print(np.max(sim, axis=1).min())
# # acc = np.array(total_acc_list[num_checkpoint:num_generated+num_checkpoint]).mean()
# # print(acc)




