
import cv2
import numpy as np
import torch
import config
from vision.lane_segmentation import LaneSegmentationModel, deeplab_predict
from vision.detect_ball_yolo import find_ball, draw_path_smooth
from vision.mask_processing import PostProcessor
from vision.perspective_transformer import BirdsEyeTransformer
from vision.trajectory import Trajectory

class ProcessFrame():

    def __init__(self, out):

        self.frame_idx = 0
        self.first_point = True
        self.out = out

        # initialize the models and classes once
        self.post_process = PostProcessor()
        self.perspective = BirdsEyeTransformer()
        self.ball_trajectory = Trajectory()

        # deeplab model setup    
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LaneSegmentationModel(n_classes=1).to(self.device)
        self.model.load_state_dict(torch.load(config.LANE_MODEL, map_location=self.device))
        self.model.eval()
        
        pass

    def _create_display(self, name, display, out=False):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, display)
        if out:
            out.write(display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return  # Exit on 'q' key  

    def process_frame(self, frame):
        frame_idx = self.frame_idx
        if frame_idx != 1 and frame_idx % config.STEP != 0: return None, None, 0

        # t_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        display = frame.copy()
                
        # predict lane
        device = self.device
        model = self.model
        pred_mask = deeplab_predict(model, device, frame.copy()) 
        if pred_mask is None: return None

        # overlay 
        mask_color = np.zeros_like(frame)
        mask_color[:, :, 1] = pred_mask  # green
        display = cv2.addWeighted(frame, 1.0, mask_color, 0.4, 0)

        # TODO: VALIDATE MASK
        # post process the lane
        post_process = self.post_process 
        extraction, mask_boundaries = post_process.apply(pred_mask.copy(), frame.copy()) 
        if extraction is None or mask_boundaries is None: return None
        left_angle, right_angle = mask_boundaries
        if left_angle is None or right_angle is None: return None



        # find the ball
        found_ball_point = find_ball(extraction.copy(), display)
        if not found_ball_point: 
            self._create_display("Lane Display", display, out=self.out) # show only segmented lane
            return None # no need for further processing
        self._create_display("Lane Display", display, out=self.out) # display segmented lane and ball

        # see birds-eye view
        perspective = self.perspective
        full_warp, M_full = perspective.transform(frame.copy(), extraction.copy(), 
                                                    left_angle, right_angle, 
                                                    alpha=2.3) 
        if full_warp is None or M_full is None: return None
        self._create_display("Birds Eye View", full_warp)

        # convert detected point to perspective transformed point
        pt = np.array(found_ball_point, dtype=np.float32).reshape(1, 1, 2)
        pt_warped = cv2.perspectiveTransform(pt, M_full)
        x_w, y_w = pt_warped[0, 0]
        first_point = self.first_point
        if first_point:
            first_point = False 
            y_w = y_w-50 # constant to increase y
            x_w = x_w-10 # constant to increase x
        else:
            y_w = y_w-170 # constant to increase y

        ball_trajectory = self.ball_trajectory
        x_smooth, y_smooth = draw_path_smooth(int(x_w), int(y_w), ball_trajectory, full_warp)  

        return x_smooth, y_smooth, 0

        