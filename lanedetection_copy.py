import cv2
from rpistream.camera import Camera
import numpy as np
import time
import colorsys
import sys
from sklearn import svm
import pickle


def normLayer(l):
    return (l-np.min(l))/(np.max(l)-np.min(l))


class ColorProfile:
    lanes = {
        "yellow": (213, 177, 50),  # rgb
        "white": (255, 255, 255),
        "grey": (150, 150, 150)
    }


horizonOffset = -100
HhorizonOffset = 100

def getDefault(img):
    height = img.shape[0]
    width = img.shape[1]
    return [
        (0, height),
        (width//2 - HhorizonOffset, height//2+horizonOffset),
        (width//2 + HhorizonOffset, height//2+horizonOffset),
        (width, height)
    ]

def unWarp(self,img, perpVerts, sizex=200, sizey=200):
    dst = np.array([[0, 0], [sizex, 0], [sizex, sizey], [0, sizey]])
    M = cv2.getPerspectiveTransform(perpVerts, dst)
    return cv2.warpPerspective(img, M)


def grayscale(img): return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, np.int32([vertices]).astype("int32"), (255, 255, 255))
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def autoCanny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


def unzero(x):
    if x == 0:
        x = 0.0001
    return x


class LaneDetector:
    def __init__(self, **kwargs):
        self.kProfile = None
        self.kLabels = None
        self.kNames = None
        self.calibrated = False
        self.clf = None

    def getCalibImage(self, cam, iters=10):
        img = None
        for i in range(iters):
            img = cam.image
        return img

    def calibrateKmeans(self, img, profile, **kwargs):

        K = kwargs.get("K", 5)
        debug = kwargs.get("debug", False)
        blurSize = kwargs.get("blurSize", (5, 5))
        w = img.shape[1]
        h = img.shape[0]
        img = cv2.GaussianBlur(img, blurSize, 0)
        Z = img.reshape((-1, 3))

        # convert to np.float32
        Z = np.float32(Z)
        kProfile = {}
        kLabels = {}
        kNames = {}
        stepSize = kwargs.get("stepSize", 5)

        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, labels, center = None, None, None
        if sys.version_info[0] == 3:
            ret, labels, center = cv2.kmeans(
                Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        else:
            ret, labels, center = cv2.kmeans(
                Z, K, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        chsv = np.array([colorsys.rgb_to_hsv(*(c[::-1]/255))
                         for c in center])  # Center colors as HSV

        Count = 0
        for name in profile:
            color = np.array(colorsys.rgb_to_hsv(
                *(np.array(profile[name])/255)))  # Profile color as HSV
            losses = np.abs(chsv-color).mean(axis=1)  # Color diffs
            n = np.argmin(losses)  # Find closest center color to profile color
            kProfile[name] = chsv[n]
            kLabels[n] = name
            kNames[name] = n

        center = np.uint8(center)

        res = center[labels.flatten()]
        res2 = res.reshape((img.shape))

        labels = labels.reshape((img.shape[0], img.shape[1]))

        trainX = []
        trainY = []
        for x in range(0, labels.shape[1], stepSize):
            for y in range(0, labels.shape[0], stepSize):
                label = labels[y, x]
                if not label in kLabels:
                    continue
                trainX.append(img[y, x])
                trainY.append(label)
        trainX, trainY = np.array(trainX), np.array(trainY)
        print(kLabels)
        print("Training size:", trainX.size)
        self.clf = svm.LinearSVC()
        self.clf.fit(trainX, trainY)

        self.kProfile = kProfile
        self.kLabels = kLabels
        self.kNames = kNames

        self.calibrated = True

        if debug:
            return res2

    def process3(self, imgin):
        #img = cv2.resize(img,(0,0),fx=0.5,fy=0.5)
        shape = imgin.shape
        pixels = shape[0]*shape[1]

        bools = (self.clf.predict(imgin.reshape(pixels, 3)).reshape(
                (shape[0], shape[1], 1)) == self.kNames["yellow"]).astype("float")

        
        return bools

    def loadSvm(self, path):
        with open(path, 'rb') as fid:
            self.clf = pickle.load(fid)

    def saveSvm(self, path):
        with open(path, 'wb') as fid:
            pickle.dump(self.clf, fid)

    def log(self, m):
        if self.verbose:
            print(m)  # printout if verbose


if __name__ == "__main__":
    cam = Camera(mirror=True)
    LD = LaneDetector()
    p = ColorProfile.lanes
    calibImg = LD.getCalibImage(cam)
    res = LD.calibrateKmeans(calibImg, p, debug=True, stepSize=10)
    with open("model.pkl","wb") as fl:
        pickle.dump(LD.clf,fl)
    # print(LD.kProfile)
    while 1:
        cv2.imshow('my webcam', LD.process3(cam.image))
        if cv2.waitKey(1) == 27:
            break  # esc to quit