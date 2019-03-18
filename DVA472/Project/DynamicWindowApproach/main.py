#import importlib
#importlib.import_module(dynamic_window_approach.py)
import numpy as np
import dynamic_window_approach as DWA
#from scipy import interpolate
#dwa = DWA()
class obstacle:
    def __init__(self):
        self.lines = [] # Number of right lines, ymin, ymax, xmin, xmax
        self.boxes = [] # Number of boxes, ymin, ymax, xmin, xmax, distance
        self.gx = 0.0
        self.gy = 5.0
        
def main():
    obs=obstacle()
    angle = 0.0
    while(True):
        #yolo..() collect obstacles in class variables
        obs.lines = [1,10,1,280,320]
        obs.boxes = [1, 1,2] #[1,15.01,16.39,0.05,0.60,14] check if zero from yolo
        y = [192,205,248]
        y = np.sort(y)
        dist = [1,2,3]
        inter = np.interp(204, y, dist)
        print("interp:",inter)


        angle = DWA.main(obs.gx, obs.gy, obs.lines, obs.boxes, angle)



if __name__ == '__main__':
    main()