import numpy as np
import argparse
import cv2
import argparse

ap = argparse.ArgumentParser() # Create argument parser 
ap.add_argument('-tl',type= float) # User input lower threshold
ap.add_argument('-th',type= float) # User input higher threshold
args = vars(ap.parse_args())
TL = args['tl'] # Save argument as variable
TH = args['th'] # Save argument as variable
cap = cv2.VideoCapture('C:\Dev\MDH\DVA472\Video_1.mp4') # Input video 
 
while(1): # Loop 
    ret, frame = cap.read() # Read frames
    gray_vid = cv2.cvtColor(frame, cv2.IMREAD_GRAYSCALE) # Change from colour to grayscale
        
    cv2.imshow('Original',frame) # Show the original frame input
    edged_frame = cv2.Canny(frame,TL,TH) # Apply canny filter
    cv2.imshow('Edges',edged_frame) # Show Canny image
    lower_yellow = np.array([20, 100, 100], dtype = 'uint8')
    upper_yellow = np.array([30, 255, 255], dtype='uint8')
                            
    mask_yellow = cv2.inRange(edged_frame, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(edged_frame, 200, 255)
    mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
    mask_yw_image = cv2.bitwise_and(edged_frame, mask_yw)
    cv2.imshow('Lane',mask_yw_image)
        
    #region_of_i = [(0,hight), (width / 2, hight / 2), (width,hight),]
                            
    if cv2.waitKey(1) & 0xFF == ord('q'): # If 'q' is pressed, end loop
        break
cap.release() 
cv2.destroyAllWindows() # close all windows
