import torch
import matplotlib.pyplot as plt
import os

INPUT_DIR = "../SynthRAD2023_SLICED/train/input"
TARGET_DIR = "../SynthRAD2023_SLICED/train/target"

files = sorted(os.listdir(INPUT_DIR))[:3]

for f in files:
    x = torch.load(os.path.join(INPUT_DIR, f))
    y = torch.load(os.path.join(TARGET_DIR, f))

    x = x.squeeze().numpy()
    y = y.squeeze().numpy()

    plt.figure(figsize=(6,3))

    plt.subplot(1,2,1)
    plt.title("MR")
    plt.imshow(x, cmap='gray')

    plt.subplot(1,2,2)
    plt.title("CT")
    plt.imshow(y, cmap='gray')

    plt.suptitle(f)
    plt.show()
