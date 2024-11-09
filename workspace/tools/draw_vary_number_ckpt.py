import sys, os
root = os.sep + os.sep.join(__file__.split(os.sep)[1:__file__.split(os.sep).index("Neural-Network-Parameter-Diffusion")+1])
sys.path.append(root)
os.chdir(root)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle


tags = ["numberckpt_001", "numberckpt_010", "numberckpt_050", "numberckpt_200", "numberckpt_300"]
caches = []
for tag in tags:
    with open(f"./results/plot_{tag}.cache", "rb") as f:
        caches.append(pickle.load(f))

org_points = [cache[-6] for cache in caches]
gen_points = [cache[-5] for cache in caches]

