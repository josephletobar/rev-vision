import cv2
import numpy as np
from vision.transformers.base_transformer import BaseTransformer
from utils.config import LANE_W, LANE_H
from math import atan2, degrees
from vision.trajectory import Trajectory

from ultralytics import YOLO
from pathlib import Path
from operator import itemgetter

buffer = Trajectory()

from dataclasses import dataclass

@dataclass
class Detection:
    label: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

class GeometricTransformer(BaseTransformer):

    def _lane_markers(self, frame):
        MODEL_PATH = Path.cwd() / "data" / "yolo_lane_markings" / "weights" / "best.pt"
        model = YOLO(MODEL_PATH)

        results = model.predict(
            source=frame,
            conf=0.25,
            imgsz=640,
            save=False
        )

        detections = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                label = f"{'arrow' if cls==0 else 'dot'}"

                d = Detection(
                    label=label,
                    confidence=conf,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2
                )
                detections.append(d)


        for d in detections:
            color = (0, 255, 0) if d.label == 'arrow' else (255, 0, 0)

            cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), color, 1)
            cv2.putText(frame, d.label, (d.x1, d.y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            
        return frame, detections
    
    def _estimation(self, frame, detections):
        dots_mid = []

        for d in detections:
            if d.label == 'dots':
                dots_mid.append( ( int((d.x1+d.x2)/2),  int((d.y1+d.y2)/2) ))
            
        

        if len(dots_mid) > 1:
            dots_mid.sort(key=itemgetter(0))
            
            (x1, y1) = dots_mid[0]
            (x2, y2) = dots_mid[-1]
    
            angle_rad = atan2(y2 - y1, x2 - x1)
            angle_deg = degrees(angle_rad)
                        
            cv2.line(frame, dots_mid[0], dots_mid[-1], (255, 0, 0), 5)

            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2

            text = f"{angle_deg:.2f}°"
            cv2.putText(frame, text, (int(mx), int(my)-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)
            
            h, w = frame.shape[:2]
            dy = np.tan(angle_rad) * w

            src = np.float32([[0,0], [w-1,0], [w-1,h-1], [0,h-1]])
            dst = np.float32([[0,0], [w-1,-dy], [w-1,h-1-dy], [0,h-1]])

            H = cv2.getPerspectiveTransform(src, dst)
            buffer.push((H, (w, h)))
            frame = cv2.warpPerspective(frame, H, (w, h))
        else:
            if buffer.last():
                frame = cv2.warpPerspective(frame, buffer.last()[0], buffer.last()[1])

        return frame
    
    def partial_transform(self, frame):
        out, detections = self._lane_markers(frame)
        out = self._estimation(out, detections)
        return out, detections

    def full_transform(self, frame, M_rel, detections):

        arrows_mid = []

        for d in detections:

            color = (0, 255, 0) if d.label == 'arrow' else (255, 0, 0)

            # top-left
            p1 = np.array([d.x1, d.y1, 1.0])
            p1 = M_rel @ p1
            p1 /= p1[2]  

            # bottom-right
            p2 = np.array([d.x2, d.y2, 1.0])
            p2 = M_rel @ p2
            p2 /= p2[2]

            x1, y1 = int(p1[0]), int(p1[1])
            x2, y2 = int(p2[0]), int(p2[1])

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            cv2.putText(frame, d.label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            
            if d.label == 'arrow':
                mx = (x1 + x2) // 2
                my = (y1 + y2) // 2
                arrows_mid.append((mx, my))

        if arrows_mid:
            avg_x = sum(p[0] for p in arrows_mid) // len(arrows_mid)
            avg_y = sum(p[1] for p in arrows_mid) // len(arrows_mid)
            arrows_avg = (avg_x, avg_y)
        else:
            arrows_avg = None

        try: 
            h, w = frame.shape[:2]
            w_mid = w//2
            cv2.line(frame, (0, avg_y), (w - 1, avg_y), (255, 255, 255), 2)

            cv2.line(frame, (w_mid, avg_y), (w_mid, 0), (255, 255, 255), 2)
            text = f"{avg_y}"
            cv2.putText(frame, text, (int(w_mid)+45, int(h//2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            
            pixels_per_foot = avg_y / 45.0 # 45 feet between arrows and pins

            px_arrow_to_end = h - avg_y
            cv2.line(frame, (w_mid, avg_y), (w_mid, h), (255, 255, 255), 2)
            text = f"{px_arrow_to_end}"
            cv2.putText(frame, text, (int(w_mid)+45, int(h-20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

            pixels_to_foul = 15.0 * pixels_per_foot
            # print(pixels_to_foul)

            missing_px = int(max(0, pixels_to_foul - px_arrow_to_end))

            frame = cv2.copyMakeBorder(
                frame,
                top=0,
                bottom=missing_px,
                left=0,
                right=0,
                borderType=cv2.BORDER_CONSTANT,
                value=(0, 0, 0)
            )
        except Exception as e:
            print(e)

        return frame
            

        
    
    # def transform(self, frame, partial=False):
    #     out, detections = self._lane_markers(frame)

    #     if partial == True:
    #         out = self._estimation(out, detections)

    #         left_lines, right_lines, _ = self._get_lines(out)

    #         for x1, y1, x2, y2 in left_lines:
    #             cv2.line(out, (x1, y1), (x2, y2), (255, 0, 0), 3)

    #         for x1, y1, x2, y2 in right_lines:
    #             cv2.line(out, (x1, y1), (x2, y2), (0, 0, 255), 3)

    #         right_result = self._average_lines(right_lines, out.shape[0])
    #         left_result = self._average_lines(left_lines, out.shape[0])

    #         if right_result is None or left_result is None:
    #             return out

    #         avg_right, right_angle = right_result
    #         avg_left, left_angle = left_result

    
    #         return out, (avg_left, avg_right)
        
    #     return out