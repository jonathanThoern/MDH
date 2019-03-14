from yolo import YOLO
import numpy as np
import cv2



cam = cv2.VideoCapture('outpy3.avi')
#cap = cv2.VideoCapture('C:\Dev\MDH\DVA472\Video_1.mp4')
yolo = YOLO()

class obstacle:
    right_lines = [] # num of right lines, y_min, y_max, x_min, x_max - add magnitude(i.e distance)???
    left_lines = [] # num of left lines, y_min, y_max, x_min, x_max - add magnitude(i.e distance)???
    boxes = [] # num of boxes, y_min, y_max, x_min, x_max, distance

    #def __init__(self):
        #self.right_lines = [] # num of right lines, y_min, y_max, x_min, x_max
        #self.left_lines = [] # num of left lines, y_min, y_max, x_min, x_max
        #self.boxes = [] # num of boxes, y_min, y_max, x_min, x_max
    
    def updateCoordinates(self, detected, scores, classes):

        temp_right = [] # num, y_min, y_max, x_min, x_max
        temp_left = [] # num, y_min, y_max, x_min, x_max
        temp_boxes = [] # num, y_min, y_max, x_min, x_max
        num_right = 0
        num_left = 0
        num_boxes = 0
        y_min = 0
        y_max = 0
        x_min = 0
        x_max = 0
        
        for i, c in reversed(list(enumerate(classes))):
            #predicted_class = self.class_names[c]
            box = detected[i]
            score = scores[i]
            y_min, x_min, y_max, x_max = box  #top, left, bottom, right = box
            y_min = max(0, np.floor(y_min + 0.5).astype('int32'))
            x_min = max(0, np.floor(x_min + 0.5).astype('int32'))
            y_max = min(416, np.floor(y_max + 0.5).astype('int32'))
            x_max = min(416, np.floor(x_max + 0.5).astype('int32'))

            if(classes[i] == 0):
                num_right = num_right + 1
                temp_right.extend((y_min, y_max, x_min, x_max))
            
            if(classes[i] == 1):
                num_left = num_left + 1
                temp_left.extend((y_min, y_max, x_min, x_max))

            if(classes[i] == 2):
                num_boxes = num_boxes + 1
                temp_boxes.extend((y_min, y_max, x_min, x_max))
        
        temp_right.insert(0, num_right) # Add number of, in first position
        temp_left.insert(0, num_left)
        temp_boxes.insert(0, num_boxes)

        self.right_lines = temp_right
        self.left_lines = temp_left
        self.boxes = temp_boxes

    def getBoxDistance():
        BOX_WIDTH = 20 # Expressed in cm
        FOCAL_LENGHT = 0.4 # Expressed in cm


def main():
    obs = obstacle()
    while True:
        ret,img=cam.read()
        detected, scores, classes = yolo.detect_image(img)
        #print("Detected: ", detected)
        #print("classes: ", classes)

        obs.updateCoordinates(detected, scores, classes)

        cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
        cv2.imshow("Frame", img)

        print("Right: ", obs.right_lines)
        print("Left: ", obs.left_lines)
        print("Box: ", obs.boxes)

        # Close with q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()







if __name__ == "__main__":
    main()


