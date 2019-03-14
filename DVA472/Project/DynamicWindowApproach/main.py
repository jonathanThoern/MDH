#import importlib
#importlib.import_module(dynamic_window_approach.py)
import dynamic_window_approach as DWA
#dwa = DWA()
class obstacle:
    def __init__(self):
        right_lines = [] # Number of right lines, ymin, ymax, xmin, xmax, distance
        left_lines = [] # Number of left lines, ymin, ymax, xmin, xmax, distance
        boxes = [] # Number of boxes, ymin, ymax, xmin, xmax, distance
        prevgoal = []
        
def main():
    obs=obstacle()
    gx = 0.0
    gy = 5.0
    while(True):
        #yolo..() collect obstacles in class variables
        obs.right_lines = [1,0.4,1,24.0,30.0,1.0]
        obs.left_lines = [1,0.4,3.3,0.2,0.8,1.0]
        obs.boxes = [2,15.01,16.39,0.05,0.60,10,15.01,16.39,0.05,0.60,14]

        DWA.main(gx, gy, obs.right_lines, obs.left_lines, obs.boxes)

if __name__ == '__main__':
    main()