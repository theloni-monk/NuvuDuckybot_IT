import cv2
from camera import Camera
import numpy as np
import time

def process(img):
    w = img.shape[1]
    h = img.shape[0]
    img=cv2.GaussianBlur(img,(5,5),0)
    img = (cv2.cvtColor(img, cv2.COLOR_BGR2HSV)/255).astype("float")
    
    # hues = img.copy()
    # hues[:,:,1:] = 0
    # hues = np.sum(hues,2)
    # vals = img.copy()
    # vals[:,:,0] = 0
    # vals[:,:,1] = 0
    # vals = np.sum(vals,2)
    img-=np.array([40/255,1,0.8])
    #img = np.abs(hues-1/38) + np.vals
    img = ((np.mean(np.abs(img),axis=2)<0.3)).astype("float")
    edges = cv2.Canny((img*255).astype("uint8"), threshold1=200, threshold2=300)
    #mask = cv2.inRange(img, lower, upper)
    #img = cv2.
    #img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_BGR2GRAY)/255
    lines = cv2.HoughLinesP(imgg, 1, np.pi/180, 180, 20, 15)
    
    
    return edges

if __name__ == "__main__":
    cam = Camera(mirror=True)
    while 1:
        cv2.imshow('my webcam', process(cam.image))
        if cv2.waitKey(1) == 27:
            break  # esc to quit