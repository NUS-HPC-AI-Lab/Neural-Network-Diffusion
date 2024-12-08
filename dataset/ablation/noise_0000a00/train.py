import os


if __name__ == "__main__":
    if not os.path.exists("../../main/cifar100_resnet18/pretrained.pth"):
        os.system(f"cd ../../main/cifar100_resnet18 && CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} python train.py")
    os.system("cp ../../main/cifar100_resnet18/pretrained.pth ./")