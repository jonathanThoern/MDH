import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
 
cap = cv2.VideoCapture('C:\Dev\MDH\DVA472\Video_1.mp4')

 
while(1):
    
    ret, frame = cap.read()
    gray_vid = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Original',frame)
    edged_frame = cv2.Canny(frame,100,200)
    cv2.imshow('Edges',edged_frame)
    lower_yellow = np.array([20, 100, 100], dtype = 'uint8')
    upper_yellow = np.array([30, 255, 255], dtype='uint8')
                            
    #mask_yellow = cv2.inRange(edged_frame, lower_yellow, upper_yellow)
    #mask_white = cv2.inRange(edged_frame, 200, 255)
    #mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
    #mask_yw_image = cv2.bitwise_and(edged_frame, mask_yw)
    cv2.imshow('Lane',mask_yw_image)

    k= cv2.waitKey(5);0xFF
    if k==27:
        break
cap.release()
cv2.destroyAllWindows()


def region_of_interest(img, vertices):
    # Define a blank matrix that matches the image height/width.
    mask = np.zeros_like(img)
    # Retrieve the number of color channels of the image.
    channel_count = img.shape[2]
    # Create a match color with the same color channel counts.
    match_mask_color = (255,) * channel_count
      
    # Fill inside the polygon
    cv2.fillPoly(mask, vertices, match_mask_color)
    
    # Returning the image only where mask pixels match
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image