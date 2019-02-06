#from transform import four_point_transform
import numpy as np
import argparse
import cv2
 
cap = cv2.VideoCapture('C:\Dev\MDH\DVA472\Video_1.mp4')
 
while(1):
    ret, frame = cap.read()
    gray_vid = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Original',frame)
    edged_frame = cv2.Canny(gray_vid,100,200)
    cv2.imshow('Edges',edged_frame)
    #cv2.imshow('Grey',gray_vid)

    k= cv2.waitKey(5);0xFF
    if k==27:
        break
cap.release()
cv2.destroyAllWindows()

cv2.cvtColor()