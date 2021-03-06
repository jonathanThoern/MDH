from yolo import YOLO
import numpy as np
import cv2

cam = cv2.VideoCapture('outpy8.avi') # 'outpy5.avi'
#cap = cv2.VideoCapture('C:\Dev\MDH\DVA472\Video_1.mp4')
yolo = YOLO()

class obstacle:
    BOX_WIDTH = float(20) # Expressed in cm
    FOCAL_LENGHT = float(18) # Expressed in cm
    DIST_MIN = 30 # Used when box is partially outside of frame
    BOX_DIST_FACTOR = 0.75 
    TWO_BOX_RATIO = 0.45
    BOX_OUT_RATIO = 0.65
    lines = [] # num of lines, y_min, y_max, x_min, x_max
    boxes = [] # num of boxes, y_dist, x_pos (in meter!)

    ''' Is not needed 
    def __init__(self):
        self.right_lines = [] # num of right lines, y_min, y_max, x_min, x_max
        self.left_lines = [] # num of left lines, y_min, y_max, x_min, x_max
        self.boxes = [] # num of boxes, y_min, y_max, x_min, x_max
    '''
    
    def updateCoordinates(self, detected, scores, classes):

        temp_lines = [] # num, y_min, y_max, x_min, x_max
        temp_boxes = [] # num of boxes, y_dist, x_pos (in meter!)
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
                #print("top, left, bottom, right", box)
                boxW = x_max - x_min
                boxH = y_max - y_min
                boxRatio = boxH / boxW
                num_boxes = num_boxes + 1

                if(boxRatio < self.TWO_BOX_RATIO): # Two Boxes are shown as one
                    num_boxes = num_boxes + 1 # add another box
                    divider = x_min + (boxW / 2)
                    box1_xmin = x_min
                    box1_xmax = divider - 1 # Separate box edges from each other
                    box2_xmin = divider + 1
                    box2_xmax = x_max
                    box1_center = box1_xmin + (box1_xmax - box1_xmin)
                    box2_center = box2_xmin + (box2_xmax - box2_xmin)
                    #print("test center: ", box1_center)
                    box1_dist = self.getBoxDistance(boxW/2)/100 # in meter

                    ### Distance from IR and Kalman could be applied here ###

                    box2_dist = box1_dist # Use same distance for now...
                    #print("test distance: ", box1_dist)
                    '''
                    if(x_min > 0 and x_max < 416): #Dont calculate on boxes out of frame
                        box1_dist = self.getBoxDistance(box1_xmax-box1_xmin)
                        box2_dist = self.getBoxDistance(box2_xmax-box2_xmin)
                    else:
                        box1_dist = self.DIST_MIN # Set fixed value when box partially out of frame
                        box2_dist = self.DIST_MIN
                    '''
                    #box1_offset = (box1_center - 208) # UNCLEAR 208 + ?
                    #box2_offset = (box2_center - 208)
                    #box1_pix_m = (box1_dist * self.BOX_DIST_FACTOR) / 416
                    #box2_pix_m = (box2_dist * self.BOX_DIST_FACTOR) / 416
                    #box1_xpos = round(box1_pix_m * box1_offset,2)
                    #box2_xpos = round(box2_pix_m * box2_offset,2)

                    temp_boxes.extend((box1_center, box1_dist)) 
                    temp_boxes.extend((box2_center, box2_dist))
                
                elif(boxRatio > self.BOX_OUT_RATIO):
                    boxW = boxH / boxRatio
                    box_dist = self.getBoxDistance(boxW) # in cm

                    ### Distance from IR and Kalman could be applied here ###

                    if(x_min > 10): # What side is outside if frame
                        box_center = x_min + (boxW/2)
                    else:
                        box_center = x_max - (boxW/2)

                    #box_offset = (box_center - 208) # UNCLEAR 208 + ?
                    #box_pix_m = (box_dist * self.BOX_DIST_FACTOR)/416
                    #box_xpos = round(box_pix_m * box_offset,2)
                    temp_boxes.extend((box_center, box_dist))

                else:
                    box_dist = self.getBoxDistance(boxW) # in cm
                    #print("test distance: ", box_dist)

                    ### Distance from IR and Kalman could be applied here ###

                    box_center = x_min + (boxW/2)
                    #print("test center: ", box_center)
                    #box_offset = (box_center - 208) # UNCLEAR 208 + ?
                    #print("test offset: ", box_offset)
                    #box_pix_m = (box_dist * self.BOX_DIST_FACTOR)/416
                    #print("test pix_m: ", box_pix_m)
                    #box_xpos = round(box_pix_m * box_offset,2)
                    #print("test xpos: ", box_xpos)
                    temp_boxes.extend((box_center, box_dist))


        #print("test num boxes: ", num_boxes)
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
        right_centers, left_centers = getCenterToLines(obs.lines, 0)
        print("right centers: ", right_centers)
        print("left centers: ", left_centers)
        print("obstacles: ", obs.boxes)

        # cv2 rectangle could be used here to show detected graphically
        cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
        cv2.imshow("Frame", img)


        #print("Lines: ", obs.lines)
        #print("Box: ", obs.boxes)

        '''
        print("Nr of Boxes: ", obs.boxes[0])
        if(obs.boxes[0]==1): # num of boxes, y_dist, x_pos (in meter!)
            print("box 1 y distance: ", obs.boxes[1])
            print("box 1 x position: ", obs.boxes[2])
        if(obs.boxes[0]==2):
            print("box 1 y distance: ", obs.boxes[1])
            print("box 1 x position: ", obs.boxes[2])
            print("box 2 y distance: ", obs.boxes[3])
            print("box 2 x position: ", obs.boxes[4])
        '''
        # Close with q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()

def getCenterToLines(d_lines, prev_angle):
    retur = 3
    right_L = []
    right_L = np.array(right_L)
    left_L = []
    left_L = np.array(left_L)
    j = 1
    add_l=0
    add_r=0
    for i in range(0, d_lines[0]):

        temp_line, retur = extract_lines(j,d_lines)
        j = j+4
        if retur == 0:
            if add_r != 0:
                right_L = np.concatenate((right_L,temp_line), axis = 0)
            else:
                right_L = temp_line
            add_r+= 1
        elif retur == 1:
            if add_l !=0:
                left_L = np.concatenate((left_L,temp_line), axis = 0) # CHANGE CONCATENATE, i!=0 galler ej
            else:
                left_L = temp_line
            add_l+= 1
        elif retur == 2:
            if prev_angle < 370:
                temp_line[add_l]=[100, 200]
                if add_l !=0:
                    left_L = np.concatenate((left_L,temp_line), axis = 0) 
                else:
                    left_L = temp_line
            else:
                temp_line[add_r] = [300, 200]
                if add_r != 0:
                    right_L = np.concatenate((right_L,temp_line), axis = 0)
                else:
                    right_L = temp_line
                add_r+= 1
    return right_L, left_L

def extract_lines(j,boxes):

    retur=2
    # Extract corners of detected lines and obstacles
    ymin=boxes[j]
    ymax=boxes[j+1]

    xmin=boxes[j+2]
    xmax=boxes[j+3]

    # Calculate middle
    y = (ymax + ymin)/2

    # Calculate middle
    x = (xmax + xmin)/2
    
    # Check if line can be determined with confidence 
    if x>228:
        retur=0 # right
    elif x<188:
        retur=1 # left

    X1,Y1 = np.meshgrid(x,y)
    ob = np.array([X1.flatten(), Y1.flatten()]).T
        
    return ob, retur

def getSteering(r_cent, l_cent, obs_data):

    if(obs_data[0] > 0):
        

if __name__ == "__main__":
    main()


