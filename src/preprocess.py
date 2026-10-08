from pathlib import Path
import torch
import yaml

RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

with open("params.yaml") as f:
    p = yaml.safe_load(f)["preprocess"]

seed = p["seed"]
val_size = p["val_size"]
torch.manual_seed(seed)

train = torch.load(RAW / "train.pt", weights_only=False)
test = torch.load(RAW / "test.pt", weights_only=False)

# NHWC uint8 -> NCHW float32 in [0, 1]
x = train["x"].permute(0, 3, 1, 2).float() / 255.0
y = train["y"].long()
x_test = test["x"].permute(0, 3, 1, 2).float() / 255.0
y_test = test["y"].long()

# CIFAR-10 standardization constants
mean = torch.tensor([0.48, 0.47, 0.45]).view(1, 3, 1, 1)
std = torch.tensor([0.24, 0.24, 0.26]).view(1, 3, 1, 1)
x = (x - mean) / std
x_test = (x_test - mean) / std

# Simple deterministic augmentation: horizontally flip every second training image.
x[::2] = torch.flip(x[::2], dims=[3])

n = len(x)
perm = torch.randperm(n)
n_val = int(n * val_size)
val_idx, train_idx = perm[:n_val], perm[n_val:]

torch.save({"x": x[train_idx], "y": y[train_idx]}, OUT / "train.pt")
torch.save({"x": x[val_idx], "y": y[val_idx]}, OUT / "val.pt")
torch.save({"x": x_test, "y": y_test}, OUT / "test.pt")

print(f"train={len(train_idx)}, val={len(val_idx)}, test={len(x_test)}")
