import numpy as np
import cv2

cap = cv2.VideoCapture(0) #Input webcam

while True:#KLoop frames
    
    ret, frame = cap.read()#Get frame
    fast = cv2.FastFeatureDetector_create(threshold = 40, nonmaxSuppression = True, type = cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)#Configure parameters for FAST
    coners_detected = fast.detect(frame,None)#Find the coners
    corners_marked = cv2.drawKeypoints(frame, corners_detected, None, color=(255,0,0))#Mark detected corners
	cv2.imshow('FAST',corners_marked)#Show the detected corners

    if cv2.waitKey(1) & 0xFF == ord('q'): # If 'q' is pressed, end loop
        break
cap.release()#Release resource - webcam
cv2.destroyAllWindows()#Close all wondows
