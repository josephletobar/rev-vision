import os
import json
import numpy as np
from PIL import Image
from labelme import utils

input_dir = "data/annotations"
output_dir = "data/masks"
os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(input_dir):
    if not file.endswith(".json"):
        continue

    json_path = os.path.join(input_dir, file)
    with open(json_path, "r") as f:
        data = json.load(f) # convert json to python dict

    image_data = utils.image.img_b64_to_arr(data["imageData"])
    label_name_to_value = {"_background_": 0} # set unlabeled pixels to 0

    # assign labeled shapes to 1
    for shape in data["shapes"]:
        label_name_to_value[shape["label"]] = 1

    label = utils.shapes_to_label(image_data.shape, data["shapes"], label_name_to_value) # convert polygon vector into 2d label image
    label = (label * 255).astype(np.uint8) # convert to a binary mask

    out_path = os.path.join(output_dir, file.replace(".json", ".png"))
    Image.fromarray(label).save(out_path) # convert numpy array to image
    print(f"Saved: {out_path}")