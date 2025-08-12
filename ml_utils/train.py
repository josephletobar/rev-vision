import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from lane_dataset import LaneDataset
from unet import UNet

device = torch.device("cpu")

# Load dataset
dataset = LaneDataset("data/images", "data/masks")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Initialize model
model = UNet().to(device)
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(50):
    model.train()
    total_loss = 0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        preds = model(images)
        loss = loss_fn(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "lane_model.pth")
print("Model saved to lane_model.pth")