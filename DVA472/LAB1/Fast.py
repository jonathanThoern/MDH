import numpy as np
import cv2
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    #cv2.imshow('Original',frame)

    # Initiate FAST object with default values
    fast = cv2.FastFeatureDetector_create(threshold = 20)

    # find and draw the keypoints
    kp = fast.detect(frame,None)
    #img2 = cv2.drawKeypoints(img, kp, None, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img2 = cv2.drawKeypoints(frame, kp, None, color=(255,0,0),flags = cv2.FAST_FEATURE_DETECTOR_NONMAX_SUPPRESSION)

    # Print all default params
    #print "Threshold: ", fast.getInt('threshold')
    #print "nonmaxSuppression: ", fast.getBool('nonmaxSuppression')
    #print "neighborhood: ", fast.getInt('type')
    #print "Total Keypoints with nonmaxSuppression: ", len(kp)

    cv2.imshow('fast_true',img2)

    # Disable nonmaxSuppression
    #fast.setBool('nonmaxSuppression',0)
    #kp = fast.detect(frame,None)

    #print "Total Keypoints without nonmaxSuppression: ", len(kp)

    #img3 = cv2.drawKeypoints(frame, kp, None, color=(255,0,0))

    #cv2.imwrite('fast_false',img3)

    k= cv2.waitKey(5);0xFF
    if k==27:
        break
cap.release()
cv2.destroyAllWindows()