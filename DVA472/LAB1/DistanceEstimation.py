import numpy as np
import cv2
import os
def distance_to_camera(knownWidth, focalLength, perWidth):
    # compute and return the distance from the image to the camera
    return (knownWidth * focalLength) / perWidth
face_cascade = cv2.CascadeClassifier(os.getcwd()+'\\haarcascade_frontalface_default.xml')
KNOWN_DISTANCE =float(input("enter known distace:"))
KNOWN_WIDTH=float(input("enter known width:"))
foclLenght=0
cam=cv2.VideoCapture(0)
ret,img=cam.read()
if ret==True:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        w=w*0.0264583333
        foclLenght=(w* KNOWN_DISTANCE) / KNOWN_WIDTH
        print("perc width",w)
        print("focal: ",foclLenght)
    cv2.imshow('img', img)
    cv2.waitKey(0)
cv2.destroyAllWindows()
while True:
    ret,img=cam.read()
    if ret==True :
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            w = w * 0.0264583333
            distance=distance_to_camera(KNOWN_WIDTH,foclLenght,w)
            cv2.putText(img, "%.2fcm" % distance,
                        (img.shape[1] - 350, img.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                        2.0, (255, 0, 0), 3)
            cv2.imshow("image",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()