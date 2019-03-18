from yolo import YOLO
import numpy as np
import cv2

cam = cv2.VideoCapture('outpy5.avi') # 'outpy5.avi'
#cap = cv2.VideoCapture('C:\Dev\MDH\DVA472\Video_1.mp4')
yolo = YOLO()

class obstacle:
    BOX_WIDTH = float(20.0) # Expressed in cm
    FOCAL_LENGHT = float(18) # Expressed in cm
    DIST_MIN = 30 # Used when box is partially outside of frame
    TWO_BOX_RATIO = 3.5
    lines = [] # num of lines, y_min, y_max, x_min, x_max
    boxes = [] # num of boxes, y_min, y_max, x_min, x_max, distance(in cm!)

    ''' Is not needed 
    def __init__(self):
        self.right_lines = [] # num of right lines, y_min, y_max, x_min, x_max
        self.left_lines = [] # num of left lines, y_min, y_max, x_min, x_max
        self.boxes = [] # num of boxes, y_min, y_max, x_min, x_max
    '''
    
    def updateCoordinates(self, detected, scores, classes):

        temp_lines = [] # num, y_min, y_max, x_min, x_max, score
        temp_boxes = [] # num, y_min, y_max, x_min, x_max
        num_lines = 0
        num_boxes = 0
        y_min = 0
        y_max = 0
        x_min = 0
        x_max = 0
        
        for i, c in reversed(list(enumerate(classes))):
            #predicted_class = self.class_names[c]
            box = detected[i]
            # score = scores[i] Not used, but could be
            y_min, x_min, y_max, x_max = box  #top, left, bottom, right = box
            y_min = max(0, np.floor(y_min + 0.5).astype('int32'))
            x_min = max(0, np.floor(x_min + 0.5).astype('int32'))
            y_max = min(416, np.floor(y_max + 0.5).astype('int32'))
            x_max = min(416, np.floor(x_max + 0.5).astype('int32'))

            if(classes[i] == 0 or classes[i]  == 1): # Line (either R or L)
                
                num_lines = num_lines + 1
                temp_lines.extend((y_min, y_max, x_min, x_max))

            if(classes[i] == 2): # Obstacle

                boxW = x_max - x_min
                boxH = y_max - y_min
                boxRatio = boxW / boxH

                if(boxRatio > self.TWO_BOX_RATIO):
                    divider = x_min + (boxW / 2)
                    box1_xmin = x_min
                    box1_xmax = divider - 1
                    box2_xmin = divider + 1
                    box2_xmax = x_max
                    if(x_min > 0 and x_max < 416): #Dont calculate on boxes out of frame
                        box1_dist = self.getBoxDistance(box1_xmax-box1_xmin)
                        box2_dist = self.getBoxDistance(box2_xmax-box2_xmin)
                    else:
                        box1_dist = self.DIST_MIN # Set fixed value when box partially out of frame
                        box2_dist = self.DIST_MIN
                    num_boxes = num_boxes + 2
                    temp_boxes.extend((y_min, y_max, box1_xmin, box1_xmax, box1_dist))
                    temp_boxes.extend((y_min, y_max, box2_xmin, box2_xmax, box2_dist))
                else:
                    num_boxes = num_boxes + 1
                    temp_boxes.extend((y_min, y_max, x_min, x_max))
                    width = x_max - x_min
                    if(x_min > 0 and x_max < 416): #Dont calculate on boxes out of frame
                        box_dist = self.getBoxDistance(width)
                        # Distance from IR and Kalman could be applied here before appending to array
                    else:
                        box_dist = self.DIST_MIN # Set fixed value when box partially out of frame
                    temp_boxes.append(box_dist)

        temp_lines.insert(0, num_lines) # Add number of, in first position
        temp_boxes.insert(0, num_boxes)

        self.lines = temp_lines
        self.boxes = temp_boxes

    def getBoxDistance(self, width):
        
        width = width * 0.0264583333
        distance = (self.BOX_WIDTH * self.FOCAL_LENGHT) / width
        distance = int(round(distance, 0))
        return distance # returns distance to obstacle in cm
            
def main():
    obs = obstacle()
    while True:
        ret,img=cam.read()
        detected, scores, classes = yolo.detect_image(img)
        obs.updateCoordinates(detected, scores, classes)

        # cv2 rectangle could be used here to show detected graphically
        cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
        cv2.imshow("Frame", img)

        print("Lines: ", obs.lines)
        print("Box: ", obs.boxes)

        # Close with q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


