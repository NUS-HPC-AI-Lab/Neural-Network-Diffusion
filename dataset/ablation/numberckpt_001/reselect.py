import os
import shutil
import random


src = [os.path.join("../numberckpt_500/checkpoint", i)
       for i in os.listdir("../numberckpt_500/checkpoint")]
os.makedirs("./checkpoint", exist_ok=True)
dst = "./checkpoint"


src.sort()
for i in src:
    shutil.copy(i, os.path.join(dst, "origin.pth"))
for i in src:
    shutil.copy(i, os.path.join(dst, "repeat.pth"))
