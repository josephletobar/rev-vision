import os
import torch
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
from unet import UNet 

# Load model 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet()  # use same init as in train.py
model.load_state_dict(torch.load("ml_utils/weights/lane_unet_model.pth", map_location=device))
model.to(device)
model.eval()

# Transform 
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
])

# Predict function
def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad(): 
        output = model(input_tensor) # run the transformed image through the model
        pred_mask = (output.squeeze() > 0.5).float().cpu().numpy() # postprocessing

    return image, pred_mask

# Run on sample images 

# Run on sample images 
if __name__ == "__main__":
    test_dir = "data/images2/"
    for filename in sorted(os.listdir(test_dir))[:10]:  # preview 10 predictions
        if not filename.endswith(".png"): continue
        img_path = os.path.join(test_dir, filename)

        img, pred = predict(img_path)

        # Set up a plot
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.title("Input Image")
        plt.imshow(img)

        plt.subplot(1, 2, 2)
        plt.title("Predicted Mask")
        plt.imshow(pred, cmap="gray")

        plt.suptitle(filename)
        plt.show()