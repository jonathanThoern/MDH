import numpy as np
import cv2
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)

while True:
    # Get frame
    ret, frame = cap.read()

    # Initiate FAST object 
    fast = cv2.FastFeatureDetector_create(threshold = 40, nonmaxSuppression = True, type = cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)

    # find and draw the keypoints
    kp = fast.detect(frame,None)
    img2 = cv2.drawKeypoints(frame, kp, None, color=(255,0,0)) #flags = cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)

    cv2.imshow('fast_true',img2)

    k= cv2.waitKey(5);0xFF
    if k==27:
        break
cap.release()
cv2.destroyAllWindows()