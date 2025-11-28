import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from arrow_dataset import ArrowDataset
from arrow_model import ArrowRegressor


# -----------------
# CONFIG
# -----------------
IMG_DIR = "data/arrow_data/images_out"
HM_DIR  = "data/arrow_data/heatmaps"
SAVE_DIR = "models/arrow_regression/weights"

BATCH_SIZE = 4
EPOCHS = 60
LR = 1e-4


# -----------------
# SETUP
# -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ENABLE AUGMENTATION HERE
dataset = ArrowDataset(IMG_DIR, HM_DIR, augment=True)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,       # keep 0 for macOS
    pin_memory=False
)

model = ArrowRegressor().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

os.makedirs(SAVE_DIR, exist_ok=True)
best_loss = float("inf")
best_path = os.path.join(SAVE_DIR, "best_arrow_model.pth")


# -----------------
# TRAINING LOOP
# -----------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for imgs, hms in loader:
        imgs = imgs.to(device)
        hms  = hms.to(device)

        preds = model(imgs)
        loss = criterion(preds, hms)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

    # save best
    if total_loss < best_loss:
        best_loss = total_loss
        torch.save(model.state_dict(), best_path)
        print(f"  Saved new best model (loss {best_loss:.4f})")


print("Training complete.")
print(f"Best model saved at: {best_path}")