import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from arrow_dataset import ArrowDataset
from arrow_model import ArrowRegressor


# config
IMG_DIR = "data/arrow_data/images_out"
HM_DIR  = "data/arrow_data/heatmaps"
SAVE_DIR = "models/arrow_regression/weights"

BATCH_SIZE = 4
EPOCHS = 200          # you can change this
LR = 1e-4


# setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = ArrowDataset(IMG_DIR, HM_DIR)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = ArrowRegressor().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

os.makedirs(SAVE_DIR, exist_ok=True)

best_loss = float("inf")
best_path = os.path.join(SAVE_DIR, "best_arrow_model.pth")


# training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for imgs, hms in loader:
        # imgs and hms are (B,3,H,W)
        imgs = imgs.to(device)
        hms  = hms.to(device)

        preds = model(imgs)
        loss = criterion(preds, hms)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

    # save best model
    if total_loss < best_loss:
        best_loss = total_loss
        torch.save(model.state_dict(), best_path)
        print(f"  Saved new best model (loss {best_loss:.4f})")

print("Training complete.")
print(f"Best model saved at: {best_path}")