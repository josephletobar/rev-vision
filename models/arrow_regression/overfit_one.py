from arrow_dataset import ArrowDataset
from arrow_model import ArrowRegressor
import torch
import torch.nn as nn
import torch.optim as optim

IMG_DIR = "data/arrow_data/images_out"
HM_DIR  = "data/arrow_data/heatmaps"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ds = ArrowDataset(IMG_DIR, HM_DIR)
img, hm = ds[0]

img = img.unsqueeze(0).to(device)
hm  = hm.unsqueeze(0).to(device)

model = ArrowRegressor().to(device)
opt   = optim.Adam(model.parameters(), lr=1e-4)
crit  = nn.MSELoss()

print("Starting overfit test on one image...")

for epoch in range(500):
    opt.zero_grad()
    pred = model(img)
    loss = crit(pred, hm)
    loss.backward()
    opt.step()

    if epoch % 25 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.6f}")