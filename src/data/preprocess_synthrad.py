import os
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm

PROCESSED_ROOT = "/usershome/cs671_user4/SynthRAD2023_Dataset"
OUTPUT_ROOT = "/usershome/cs671_user4/SynthRAD2023_SLICED"
MIN_STD = 0.01


def load_pair(group, root):
    mr_row = group[group["modality"] == "mr"]
    ct_row = group[group["modality"] == "ct"]

    if mr_row.empty or ct_row.empty:
        return None, None

    mr_path = os.path.join(root, mr_row["pt_path"].values[0])
    ct_path = os.path.join(root, ct_row["pt_path"].values[0])

    mr = torch.load(mr_path)  # [1, D, H, W]
    ct = torch.load(ct_path)

    return mr.squeeze(0), ct.squeeze(0)  # → [D, H, W]


def valid_slice(x):
    return x.std().item() > MIN_STD


def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def main():
    manifest_path = os.path.join(PROCESSED_ROOT, "manifest.csv")
    df = pd.read_csv(manifest_path)

    # ✅ only brain
    df = df[df["region"] == "brain"]

    grouped = df.groupby("patient_id")

    total_saved = 0

    for pid, group in tqdm(grouped, desc="Patients"):
        split = group["split"].iloc[0]

        mr, ct = load_pair(group, PROCESSED_ROOT)
        if mr is None:
            continue

        # 🔥 AXIS 1 slicing (CORRECT)
        mr_valid = [i for i in range(mr.shape[1]) if mr[:, i, :].std().item() > MIN_STD]
        ct_valid = [i for i in range(ct.shape[1]) if ct[:, i, :].std().item() > MIN_STD]

        if len(mr_valid) == 0 or len(ct_valid) == 0:
            continue

        min_len = min(len(mr_valid), len(ct_valid))

        for k in range(min_len):
            mr_idx = mr_valid[k]
            ct_idx = ct_valid[k]

            # ✅ slice along AXIS 1
            x = mr[:, mr_idx, :]
            y = ct[:, ct_idx, :]

            x = x.unsqueeze(0)
            y = y.unsqueeze(0)

            # normalize
            x = normalize(x)
            y = normalize(y)

            input_dir = Path(OUTPUT_ROOT) / split / "input"
            target_dir = Path(OUTPUT_ROOT) / split / "target"

            input_dir.mkdir(parents=True, exist_ok=True)
            target_dir.mkdir(parents=True, exist_ok=True)

            name = f"{pid}_{k:03d}.pt"

            torch.save(x.contiguous(), input_dir / name)
            torch.save(y.contiguous(), target_dir / name)

            total_saved += 1

        del mr, ct

    print(f"\n✅ Done. Total slices saved: {total_saved}")


if __name__ == "__main__":
    main()
