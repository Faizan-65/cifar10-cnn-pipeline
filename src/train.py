from pathlib import Path
import csv
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

Path("models").mkdir(exist_ok=True)

with open("params.yaml") as f:
    p = yaml.safe_load(f)["train"]

F = p["num_filters"]
dropout = p["dropout_rate"]
lr = p["learning_rate"]
epochs = p["epochs"]
batch_size = p["batch_size"]

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print("device:", device)

train = torch.load("data/processed/train.pt", weights_only=False)
val = torch.load("data/processed/val.pt", weights_only=False)

train_loader = DataLoader(TensorDataset(train["x"], train["y"]), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(val["x"], val["y"]), batch_size=batch_size)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, F, 3, padding=1), nn.BatchNorm2d(F), nn.ReLU(),
            nn.Conv2d(F, F, 3, padding=1), nn.BatchNorm2d(F), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(F, F * 2, 3, padding=1), nn.BatchNorm2d(F * 2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(F * 2, F * 4, 3, padding=1), nn.BatchNorm2d(F * 4), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(F * 4 * 4 * 4, 512), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

history = []
for epoch in range(1, epochs + 1):
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(xb)

    model.eval()
    correct = total = 0
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            val_loss += criterion(logits, yb).item() * len(xb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)

    row = {
        "epoch": epoch,
        "train_loss": train_loss / len(train_loader.dataset),
        "val_loss": val_loss / len(val_loader.dataset),
        "val_accuracy": correct / total,
    }
    history.append(row)
    print(row)

# Save both architecture parameters and weights.
torch.save({"state_dict": model.state_dict(), "num_filters": F, "dropout_rate": dropout}, "models/model.pth")

with open("models/history.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=history[0].keys())
    writer.writeheader()
    writer.writerows(history)
