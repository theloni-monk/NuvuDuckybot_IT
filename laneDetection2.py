import cv2
from rpistream.camera import Camera
import numpy as np
import time
import colorsys
import sys
import cPickle

def normLayer(l):
    return (l-np.min(l))/(np.max(l)-np.min(l))

class ColorProfile:
    lanes = {
        "yellow":(213,177,50), #rgb
        "white":(255,255,255),
        "grey":(150,150,150)
    }

class LaneDetector:
    def __init__(self, **kwargs):
        self.kProfile = None
        self.kLabels = None
        self.calibrated = False
        self.Svm=None

    def getCalibImage(self,cam,iters=10):
        img = None
        for i in range(iters):
            img = cam.image
        return img

    def calibrateKmeans(self, img, profile, **kwargs):
        
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
            ret,labels,center=cv2.kmeans(Z,K,None, criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        else:
            ret,labels,center=cv2.kmeans(Z,K,criteria,10, cv2.KMEANS_RANDOM_CENTERS)

        chsv = np.array([colorsys.rgb_to_hsv(*(c[::-1]/255)) for c in center]) # Center colors as HSV

        Count=0
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

    def loadSvm(self, path):
        with open(path, 'rb') as fid:
            self.Svm=Cpickle.load(fid)
    
    def saveSvm(self, path):
        with open(path, 'wb') as fid:
            cPickle.dump(self.Svm, fid)  
    
    def trainSvm(self):
        


if __name__ == "__main__":
    cam = Camera(mirror=True)
    LD=LaneDetector()
    p=ColorProfile.lanes
    calibImg = LD.getCalibImage(cam)
    res=LD.calibrateKmeans(calibImg, p, debug=True)
    #print(LD.kProfile)
    while 1:
        cv2.imshow('my webcam', LD.calibrateKmeans(cam.image))
        if cv2.waitKey(1) == 27:
            break  # esc to quit