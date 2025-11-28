import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from arrow_dataset import ArrowXYDataset
from arrow_model import ArrowRegressor


IMG_DIR = "data/arrow_data/images_out"
LABEL_DIR = "data/arrow_data/labels_xy"
SAVE_DIR = "data/arrow_regression/weights_xy"

BATCH_SIZE = 4
EPOCHS = 60
LR = 1e-4

os.makedirs(SAVE_DIR, exist_ok=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = ArrowXYDataset(IMG_DIR, LABEL_DIR)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = ArrowRegressor().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

best_loss = float("inf")
best_path = os.path.join(SAVE_DIR, "best_resnet_xy.pth")


for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for imgs, coords in loader:
        imgs = imgs.to(device)
        coords = coords.to(device)

        preds = model(imgs)     # (B,6)
        loss = criterion(preds, coords)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

    if total_loss < best_loss:
        best_loss = total_loss
        torch.save(model.state_dict(), best_path)
        print(f"  Saved new best (loss {best_loss:.4f})")


print("Training complete.")
print("Best model saved at:", best_path)
