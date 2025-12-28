from vision.mask_processing import OverlayProcessor, ExtractProcessor, extraction_validator
from vision.geometric_validation import validate
from models.lane_segmentation.deeplab_predict import deeplab_predict
overlay = OverlayProcessor()
extract = ExtractProcessor()
import os
import cv2

weights = "data/weights/lane_deeplab_model_2.pth" 

PATH = "data/extracted_frames"



def main():

    # folders = [
    #     name for name in os.listdir(PATH)
    #     if os.path.isdir(os.path.join(PATH, name))
    # ]
    # print(folders)

    # for folder in os.listdir(PATH):
    #     folder_path = os.path.join(PATH, folder)
    #     print(folder_path)

        # if not os.path.isdir(folder_path):
        #     continue

    for file in os.listdir(PATH):
        file_path = os.path.join(PATH, file)
        print(file_path)

        if not file.lower().endswith(".png"):
            continue

        img = cv2.imread(file_path)

        # predict lane
        _, pred_mask = deeplab_predict(img, weights) 

        # extract the lane
        extraction = extract.apply(pred_mask, img) 

        # cv2.imshow(file_path, extraction)
        # if cv2.waitKey(30) & 0xFF == ord("q"):
        #     cv2.destroyAllWindows()
        #     return

        OUTPUT_DIR = "data/ball_detection/ball_img_extracted_json"
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        out_path = os.path.join(OUTPUT_DIR, file)
        cv2.imwrite(out_path, extraction)
main()