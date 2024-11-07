import os
import shutil
import random


src = [os.path.join("../numberckpt_500/checkpoint", i)
       for i in os.listdir("../numberckpt_500/checkpoint")]
os.makedirs("./checkpoint", exist_ok=True)
dst = "./checkpoint"


random.shuffle(src)
src = src[:200]
for i in src:
    shutil.copy(i, dst)
