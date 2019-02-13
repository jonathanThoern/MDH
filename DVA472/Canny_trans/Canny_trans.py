import numpy as np
import argparse
import cv2
import argparse

ap = argparse.ArgumentParser() 
ap.add_argument('-tl',type= float)
ap.add_argument('-th',type= float)
args = vars(ap.parse_args())
TL = args['tl']
TH = args['th']
cap = cv2.VideoCapture('/home/odroid/Desktop/Canny/Video_1.mp4')
 
while(1):
	ret, frame = cap.read()
	Gfilter = cv2.GaussianBlur(frame,(5,5), 0)
	gray_vid = cv2.cvtColor(Gfilter, cv2.COLOR_BGR2GRAY)
	cv2.imshow('Original',frame)
	canny = cv2.Canny(gray_vid,TL,TH)
	cv2.imshow('Edges',canny)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release() 
cv2.destroyAllWindows()
