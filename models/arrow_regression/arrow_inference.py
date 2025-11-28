import torch
import cv2
import numpy as np
from arrow_model import ArrowRegressor

class ArrowPredictorXY:
    def __init__(self, weight_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ArrowRegressor().to(self.device)
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.model.eval()

        self.W = 384
        self.H = 512

    def preprocess(self, img):
        # Resize to training dims
        img_resized = cv2.resize(img, (self.W, self.H))

        # Convert to float
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_rgb = img_rgb.astype(np.float32) / 255.0

        # HWC → CHW
        tensor = torch.from_numpy(img_rgb).permute(2,0,1).unsqueeze(0)
        return tensor.to(self.device)

    def predict(self, img):
        x = self.preprocess(img)

        with torch.no_grad():
            out = self.model(x)[0].cpu().numpy()  # shape (6,)

        # unpack into (x,y) triplets
        pts = out.reshape(3,2)

        return pts

if __name__ == "__main__":
    predictor = ArrowPredictorXY("models/arrow_regression/weights/best_xy_model.pth")

    img = cv2.imread("data/images2/frame_0021.png")

    orig_h, orig_w = img.shape[:2]

    # scale up from network space to original image
    scale_x = orig_w / 384
    scale_y = orig_h / 512

    pts = predictor.predict(img)

    print("Predicted points (network space):")
    print(pts)

    # draw points
    vis = img.copy()
    for (x, y) in pts:
        X = int(x * scale_x)
        Y = int(y * scale_y)
        cv2.circle(vis, (X, Y), 8, (0,0,255), 3)

    cv2.imshow("pred", vis)
    cv2.waitKey(0)