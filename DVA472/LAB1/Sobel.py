import cv2
import numpy as np
 

cap = cv2.VideoCapture('/home/odroid/Desktop/Sobel/Video_1.mp4')#Input video
while(1):#Loop over frames
	ret, frame = cap.read()#Read frame
	gray_vid = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#Coverts frame to grayscale
	cv2.imshow('Original',gray_vid)#Show grayscale frame
	sobel_horizontal_1 = cv2.Sobel(gray_vid,cv2.CV_64F,1,0,ksize = 5)#Sobel horizontal
	sobel_horizontal = cv2.convertScaleAbs(sobel_horizontal_1, alpha=8/sobel_horizontal_1.max(), beta=0.)#Covert to the absolute value
	sobel_vertical_2 = cv2.Sobel(gray_vid,cv2.CV_64F,0,1,ksize=5)#Sobel vertical
	sobel_vertical = cv2.convertScaleAbs(sobel_vertical_2, alpha=256/sobel_vertical_2.max(), beta=0.)#Covert to the absolute value
	merged = cv2.addWeighted(sobel_vertical, 0.5, sobel_horizontal, 0.5, 0, 0)#merge horizontal and vertical 
	cv2.imshow('Merged Sobel',merged)# Show the merged - Sobel output
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()#Release resource - videostream
cv2.destroyAllWindows() #Close all open windows
