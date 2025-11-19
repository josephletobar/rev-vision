import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ml_utils.lane_dataset import LaneDataset
from ml_utils.deeplab_model import get_deeplab_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
dataset = LaneDataset("data/images", "data/lane_masks")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Model (binary head inside your getter)
model = get_deeplab_model()

# LOSS:
# If your getter includes Sigmoid in the head -> use BCELoss
# If you removed Sigmoid (logits) -> use BCEWithLogitsLoss
loss_fn = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Train
for epoch in range(50):
    model.train()
    total_loss = 0.0

    for images, masks in dataloader:
        images = images.to(device)
        if masks.ndim == 3:
            masks = masks.unsqueeze(1)  # [B,1,H,W]
        masks = masks.float().to(device)

        preds = model(images)["out"] if isinstance(model(images), dict) else model(images)

        loss = loss_fn(preds, masks)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "lane_model.pth")
print("Model saved to lane_model.pth")