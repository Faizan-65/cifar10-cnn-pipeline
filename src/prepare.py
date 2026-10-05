from pathlib import Path
import torch
from torchvision.datasets import CIFAR10

RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)

train = CIFAR10(root="data/downloads", train=True, download=True)
test = CIFAR10(root="data/downloads", train=False, download=True)

torch.save({"x": torch.tensor(train.data), "y": torch.tensor(train.targets)}, RAW / "train.pt")
torch.save({"x": torch.tensor(test.data), "y": torch.tensor(test.targets)}, RAW / "test.pt")

print(f"Saved raw train: {len(train)} samples")
print(f"Saved raw test:  {len(test)} samples")
