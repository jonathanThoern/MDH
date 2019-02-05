import cv2
import numpy as np
 
cap = cv2.VideoCapture('C:\Dev\MDH\DVA472\Video_1.mp4') 

while(1):
    ret, frame = cap.read()

    greyFrame = cv2.cvtColor(frame,cv2.IMREAD_GRAYSCALE)
    #rows,cols,_ = greyFrame.shape
 
    sobel_horizontal = cv2.Sobel(greyFrame,cv2.CV_16S,1,0)
    sobel_vertical = cv2.Sobel(greyFrame,cv2.CV_16S,0,1)
 
    cv2.imshow('Original',greyFrame)
    cv2.imshow('Sobel Horizontal Filter',sobel_horizontal)
    cv2.imshow('Sobel Vertical Filter',sobel_vertical)
 
    cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()