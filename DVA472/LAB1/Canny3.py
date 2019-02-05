import numpy as np
import argparse
import cv2
 
cap = cv2.VideoCapture('C:\Dev\MDH\DVA472\Video_1.mp4')
 
while(1):
    ret, frame = cap.read()
    gray_vid = cv2.cvtColor(frame, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('Original',frame)
    edged_frame = cv2.Canny(frame,100,200)
    cv2.imshow('Edges',edged_frame)
    lower_yellow = np.array([20, 100, 100], dtype = 'uint8')
    upper_yellow = np.array([30, 255, 255], dtype='uint8')
                            
    mask_yellow = cv2.inRange(edged_frame, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(edged_frame, 200, 255)
    mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
    mask_yw_image = cv2.bitwise_and(edged_frame, mask_yw)
    cv2.imshow('Lane',mask_yw_image)
    
    k= cv2.waitKey(5);0xFF
    if k==27:
        break
cap.release()
cv2.destroyAllWindows()