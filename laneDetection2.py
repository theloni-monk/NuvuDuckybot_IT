import cv2
from rpistream.camera import Camera
import numpy as np
import time
import colorsys

class Profile:
    lanes = {
        "yellow":(213,177,91),
        "white":(255,255,255),
        "grey":(150,150,150)
    }

def calibrate(img,profile,**kwargs):
    K = kwargs.get("K",3)
    blurSize = kwargs.get("blurSize",(5,5))
    w = img.shape[1]
    h = img.shape[0]
    img=cv2.GaussianBlur(img,blurSize,0)
    Z = img.reshape((-1,3))

    # convert to np.float32
    Z = np.float32(Z)
    newProfile = {}
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    
    chsv = np.array([colorsys.rgb_to_hsv(*(c/255)) for c in center])

    for name in profile:
        color = np.array(colorsys.rgb_to_hsv(*np.array(profile[name])/255))
        losses = np.abs(chsv-color).mean(axis=1)
        n = np.argmin(losses)
        newProfile[name] = center[n]

    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    return newProfile,res2

def process(img):
    return img

def getCalibImage(cam,iters=10):
    img = None
    for i in range(iters):
        img = cam.image
    return img

if __name__ == "__main__":
    cam = Camera(mirror=True)
    calibImg = getCalibImage(cam)
    profile,res = calibrate(calibImg,Profile.lanes)
    print(profile)
    while 1:
        cv2.imshow('my webcam', res)
        if cv2.waitKey(1) == 27:
            break  # esc to quit