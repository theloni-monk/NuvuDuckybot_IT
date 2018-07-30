import cv2
from rpistream.camera import Camera
import numpy as np
import time

def calibrate(img,profile,**kwargs):
    K = kwargs.get("K",5)
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

    for name in profile:
        color = profile[name]
        losses = np.abs(center-np.array(color)).mean(axis=1)
        n = np.argmin(losses)
        newProfile[name] = center[n]
        
    return newProfile
def process(img):
    w = img.shape[1]
    h = img.shape[0]
    img=cv2.GaussianBlur(img,(5,5),0)
    Z = img.reshape((-1,3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 5
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    print("RET: ",ret)
    print("LBL: ",label.shape)
    print("CTR: ",center)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    return res2


if __name__ == "__main__":
    cam = Camera(mirror=True)
    while 1:
        cv2.imshow('my webcam', process(cam.image))
        if cv2.waitKey(1) == 27:
            break  # esc to quit