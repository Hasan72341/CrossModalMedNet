import torch
import pandas as pd
import os
import matplotlib.pyplot as plt

df = pd.read_csv("manifest.csv")
df = df[df["region"] == "brain"]

pid = df["patient_id"].iloc[0]
group = df[df["patient_id"] == pid]

mr_path = os.path.join(".", group[group["modality"]=="mr"]["pt_path"].values[0])

mr = torch.load(mr_path).squeeze()

print("Shape:", mr.shape)

for axis in range(3):
    idx = mr.shape[axis] // 2

    if axis == 0:
        img = mr[idx, :, :]
    elif axis == 1:
        img = mr[:, idx, :]
    else:
        img = mr[:, :, idx]

    plt.figure()
    plt.title(f"Axis {axis}")
    plt.imshow(img, cmap='gray')
    plt.show()
