from pathlib import Path
import json
import torch
import torch.nn as nn
import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader, TensorDataset

Path("plots").mkdir(exist_ok=True)

with open("params.yaml") as f:
    train_params = yaml.safe_load(f)["train"]

checkpoint = torch.load("models/model.pth", map_location="cpu", weights_only=False)
F = checkpoint["num_filters"]
dropout = checkpoint["dropout_rate"]

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

model = CNN()
model.load_state_dict(checkpoint["state_dict"])
model.eval()

test = torch.load("data/processed/test.pt", weights_only=False)
loader = DataLoader(TensorDataset(test["x"], test["y"]), batch_size=train_params["batch_size"])
criterion = nn.CrossEntropyLoss()

loss_sum = 0.0
correct = 0
y_true, y_pred = [], []

with torch.no_grad():
    for xb, yb in loader:
        logits = model(xb)
        loss_sum += criterion(logits, yb).item() * len(xb)
        pred = logits.argmax(1)
        correct += (pred == yb).sum().item()
        y_true.extend(yb.tolist())
        y_pred.extend(pred.tolist())

metrics = {
    "test_loss": loss_sum / len(loader.dataset),
    "test_accuracy": correct / len(loader.dataset)
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

cm = confusion_matrix(y_true, y_pred)
ConfusionMatrixDisplay(cm).plot(cmap="Blues", xticks_rotation=45)
plt.tight_layout()
plt.savefig("plots/confusion_matrix.png", dpi=150)
plt.close()

print(metrics)
