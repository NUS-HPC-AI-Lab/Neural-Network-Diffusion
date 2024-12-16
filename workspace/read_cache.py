import sys, os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# read the cache
cache_file = '../plot_save_lr00003.cache'
if not os.path.exists(cache_file):
    raise FileNotFoundError(f"Cache file {cache_file} not found")

with open(cache_file, "rb") as f:
    data = pickle.load(f)


summary = data[0]
iou_matrix = data[0]['iou_matrix']

similarity_list = np.zeros(1401)

# original model
for i in range(300):
    # obtain the max iou with the remaining 299 models
    other_models = np.concatenate([iou_matrix[0:i], iou_matrix[i+1:300]])  # 排除第i个模型
    similarity_list[i] = other_models[:, i].max()  # 取最大值


similarity_list[300:] = iou_matrix[0:300, 300:].max(axis=0)

acc_list = data[0]['total_acc_list']


original = slice(0, 301)  # original models
p_diff = slice(301, 501)  # generated models
noise_0001 = slice(501, 801)  # noise=0.001
noise_005 = slice(801, 1101)  # noise=0.05
noise_015 = slice(1101, 1401)  # noise=0.15


plt.figure(figsize=(8, 6))

# plot
plt.scatter(similarity_list[original], acc_list[original], 
           label='original', alpha=0.6, s=20)
plt.scatter(similarity_list[p_diff], acc_list[p_diff], 
           label='p-diff', alpha=0.6, s=20)
plt.scatter(similarity_list[noise_0001], acc_list[noise_0001], 
           label='noise=0.001', alpha=0.6, s=20)
plt.scatter(similarity_list[noise_005], acc_list[noise_005], 
           label='noise=0.05', alpha=0.6, s=20)
plt.scatter(similarity_list[noise_015], acc_list[noise_015], 
           label='noise=0.15', alpha=0.6, s=20)

# fig settings
plt.xlabel('maximum similarity')
plt.ylabel('accuracy (%)')
plt.title('(a) lr = 0.0003')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)

# y range
plt.ylim(0.74, 0.78)

plt.tight_layout()
plt.savefig('plot_save_lr00003.png', dpi=300, bbox_inches='tight')
plt.show()