import os
import shutil
import random


src = [os.path.join("../numberckpt_391/checkpoint", i)
       for i in os.listdir("../numberckpt_391/checkpoint")]
os.makedirs("./checkpoint", exist_ok=True)
dst = "./checkpoint"


src.sort()
src = src[:10]
for i in src:
    shutil.copy(i, dst)
