import cv2
from rpistream.camera import Camera
import numpy as np
import time
import colorsys
import sys

def normLayer(l):
    return (l-np.min(l))/(np.max(l)-np.min(l))
class ColorProfile:
    lanes = {
        "yellow":(213,177,50),
        "white":(255,255,255),
        "grey":(150,150,150)
    }
    
class LaneDetector:
    def __init__(self, **kwargs):
        self.kProfile = None
        self.kLabels = None
        self.calibrated = False

    def calibrate(self, img, profile, **kwargs):
        K = kwargs.get("K",5)
        debug=kwargs.get("debug",False)
        blurSize = kwargs.get("blurSize",(5,5))
        w = img.shape[1]
        h = img.shape[0]
        img=cv2.GaussianBlur(img,blurSize,0)
        Z = img.reshape((-1,3))

        # convert to np.float32
        Z = np.float32(Z)
        kProfile = {}

        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret,labels,center=None,None,None
        if sys.version_info[0] == 3:
            ret,labels,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        else:
            ret,labels,center=cv2.kmeans(Z,K,criteria,10)

        chsv = np.array([colorsys.rgb_to_hsv(*(c[::-1]/255)) for c in center]) # Center colors as HSV

        for name in profile:
            color = np.array(colorsys.rgb_to_hsv(*(np.array(profile[name])/255))) # Profile color as HSV
            losses = np.abs(chsv-color).mean(axis=1) # Color diffs
            n = np.argmin(losses) # Find closest center color to profile color

            kProfile[name] = chsv[n]
            print(name,chsv[n])

        center = np.uint8(center)
        res = center[labels.flatten()]
        res2 = res.reshape((img.shape))

        self.kProfile = kProfile
        self.calibrated = True
        if debug:
            return res2
    
    def process2(self,img):
        img=cv2.GaussianBlur(img,(5,5),0)
        Z = img.reshape((-1,3))

        # convert to np.float32
        Z = np.float32(Z)
        kProfile = {}
        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret,label,center=cv2.kmeans(Z,K,self.kLabels,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    
    def process(self,img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)/255
        l = [] # Losses for grey, yellow, white channels
        for name in self.kProfile:
            color = np.array(self.kProfile[name])
            losses = np.power(hsv-color,2).mean(axis=2) # Find color diffs
            l.append(losses.reshape(img.shape[0],img.shape[1],1)) # Add to losses
        l = np.concatenate((l[0], l[1], l[2]),axis=2) # Reshape losses into RGB channels
        l = np.argmin(l,2) # Find the lowest loss-ing channel for each pixel
        return (l==1).astype("float") # Find the pixels where channel 0 is the lowest loss

    def getCalibImage(self,cam,iters=10):
        img = None
        for i in range(iters):
            img = cam.image
        return img

if __name__ == "__main__":
    cam = Camera(mirror=True)
    LD=LaneDetector()
    p=ColorProfile.lanes
    calibImg = LD.getCalibImage(cam)
    res=LD.calibrate(calibImg, p, debug=True)
    #print(LD.kProfile)
    while 1:
        cv2.imshow('my webcam', LD.process(cam.image))
        if cv2.waitKey(1) == 27:
            break  # esc to quit