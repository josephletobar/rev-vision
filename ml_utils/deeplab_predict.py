import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
from ml_utils.deeplab_model import get_deeplab_model 

# Load model 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(weights):
    model = get_deeplab_model(device=device)  # initialize DeepLab model
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model

# Transform 
transform = T.Compose([
    T.Resize((960, 720)),
    T.ToTensor(),
])

# Predict function
def deeplab_predict(input_data, weights):
    model = load_model(weights)

    if isinstance(input_data, str):  # filepath
        image = Image.open(input_data).convert("RGB")
    elif isinstance(input_data, np.ndarray):  # numpy array (BGR from OpenCV)
        image = Image.fromarray(input_data[..., ::-1])  # convert BGR to RGB
    else:
        raise TypeError("Input must be a filepath or a numpy array")

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad(): 
        output = model(input_tensor)['out'] # run the transformed image through the model 
                                            # DeepLab returns dict; use ['out']. UNet returns tensor directly.
        
        # Convert model output to a binary NumPy mask:
        # 1) squeeze: remove extra batch/channel dims [1,1,H,W] → [H,W]
        # 2) > 0.5: threshold probabilities at 0.5; boolean mask
        # 3) float(): convert booleans to 0.0 / 1.0
        # 4) cpu(): move from GPU to CPU
        # 5) numpy(): convert to NumPy array (float32, values in {0.0, 1.0})
        pred_mask = (output.squeeze() > 0.5).float().cpu().numpy() 

    return image, pred_mask

# Run on sample images 
if __name__ == "__main__":
    test_dir = "data/images/"
    weights = "lane_deeplab_model"
    for filename in sorted(os.listdir(test_dir))[:50]:  # preview x predictions
        if not filename.endswith(".png"): continue
        img_path = os.path.join(test_dir, filename)

        img, pred = deeplab_predict(img_path, weights)

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