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
cap = cv2.VideoCapture('/home/odroid/Desktop/Canny/Video_1.mp4') # Input video 
 
while(1): # Loop 
    ret, frame = cap.read() # Read frames
    gray_vid = cv2.cvtColor(frame, cv2.IMREAD_GRAYSCALE) # Change from colour to grayscale
        
    cv2.imshow('Original',frame) # Show the original frame input
    edged_frame = cv2.Canny(frame,TL,TH) # Apply canny filter
    cv2.imshow('Edges',edged_frame) # Show Canny image
       
    #region_of_i = [(0,hight), (width / 2, hight / 2), (width,hight),]
                            
    if cv2.waitKey(1) & 0xFF == ord('q'): # If 'q' is pressed, end loop
        break
cap.release() 
cv2.destroyAllWindows() # close all windows
