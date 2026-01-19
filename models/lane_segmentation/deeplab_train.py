import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.lane_segmentation.lane_dataset import LaneDataset
from models.lane_segmentation.deeplab_model import get_deeplab_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
dataset = LaneDataset("data/lane_segmentation_new/images", "data/lane_segmentation_new/masks")
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Model (binary head inside your getter)
model = get_deeplab_model()
model = model.to(device)

model = get_deeplab_model().to(device)

for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        m.eval()


# LOSS:
# If your getter includes Sigmoid in the head -> use BCELoss
# If you removed Sigmoid (logits) -> use BCEWithLogitsLoss
loss_fn = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=1e-4)

torch.cuda.empty_cache()


# Train
for epoch in range(50):
    model.train()
    total_loss = 0.0

    for images, masks in dataloader:
        images = images.to(device)
        if masks.ndim == 3:
            masks = masks.unsqueeze(1)  # [B,1,H,W]
        masks = masks.float().to(device)

        out = model(images)
        preds = out["out"] if isinstance(out, dict) else out


        loss = loss_fn(preds, masks)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "lane_model.pth")
print("Model saved to lane_model.pth")