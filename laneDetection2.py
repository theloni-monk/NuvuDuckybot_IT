import cv2
from camera import Camera
import numpy as np
import time

def process(img):
    w = img.shape[1]
    h = img.shape[0]
    img=cv2.GaussianBlur(img,(3,3),0)
    img = (cv2.cvtColor(img, cv2.COLOR_BGR2HSV)/255).astype("float")
    # hues = img.copy()
    # hues[:,:,1:] = 0
    # hues = np.sum(hues,2)
    # vals = img.copy()
    # vals[:,:,0] = 0
    # vals[:,:,1] = 0
    # vals = np.sum(vals,2)
    img-=np.array([35/255,0.8,0.8])
    #img = np.abs(hues-1/38) + np.vals
    img = np.mean(np.abs(img),axis=2)
    print(img)
    #mask = cv2.inRange(img, lower, upper)
    #img = cv2.
    #img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_BGR2GRAY)/255
    img = img<0.1
    
    return img.astype("float")

if __name__ == "__main__":
    cam = Camera(mirror=True)
    while 1:
        cv2.imshow('my webcam', process(cam.image))
        if cv2.waitKey(1) == 27:
            break  # esc to quit