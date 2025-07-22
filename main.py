import numpy as np
import cv2
from lane_finder import draw_lines

cap = cv2.VideoCapture('bowling.mp4')

while cap.isOpened():
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    

     # Draw lane lines on image
    try:
        lane_detection = draw_lines(frame)
    except:
        lane_detection = frame
 
    cv2.imshow('frame', lane_detection)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()