# Hello, I wrote most of the code in this file,
# and most of the code here is unreadable and poorly documented
# This means that if something breaks or doesn't work as intended,
# you shouldn't come tell me about it, because it's probably not a bug, just an unintended feature

import cv2
from rpistream.camera import Camera
import numpy as np
import time
import colorsys
import sys
from sklearn import svm
import pickle


# LD helpers
def normLayer(l):
    return (l-np.min(l))/(np.max(l)-np.min(l))


def getDefault(h, w):
    height = h
    width = w
    hLength = 50
    hDepth = 300
    p = np.array([
        (0, height),
        (width//2 - hLength, height-hDepth),
        (width//2 + hLength, height-hDepth),
        (width, height)
    ], np.float32)
    return p


def drawVertical(img, n, color):
    cv2.line(img, (n, 0), (n, img.shape[0]), color, 2)


def unwarp(img):
    width = img.shape[1]
    height = img.shape[0]
    hLength = 45
    hDepth = 300

    # TODO: tune this
    #   no -Ian
    p = np.array([
        (0, height),
        (width//2 - hLength, height-hDepth),
        (width//2 + hLength, height-hDepth),
        (width, height)
    ], np.float32)
    dst = np.array([[0, height], [0, 0], [width, 0],
                    [width, height]], np.float32)
    M = cv2.getPerspectiveTransform(p, dst)

    return cv2.warpPerspective(img, M, (width, height))


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


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


def denoise(imgin, boolimg):
    boolimg = boolimg.astype("uint8")*255

    Cimg = grayscale(np.bitwise_and(imgin, boolimg))  # masking
    # TODO: use boolimg as a mask on normal img then threshold the img to get rid of noise

    Cimg[Cimg < 180] = 0
    Cimg = Cimg.astype("float")/255

    return Cimg


class ColorProfile:
    lanes = {
        "yellow": (200, 177, 0),  # rgb
        "white": (255, 255, 255),
        "grey": (90, 90, 70)
    }


class LaneDetector:
    def __init__(self, **kwargs):
        self.kProfile = {}
        self.kLabels = {}
        self.kNames = {}
        self.kProfRGB = {}
        self.calibrated = False
        self.clf = None

        self.stacks = None

        # Modifying while in the control loop may cause problems
        self.RAlookback = kwargs.get("RAlookback", 5)

    # calibration and svm training
    def getCalibImage(self, cam, iters=10):
        img = None
        for i in range(iters):
            img = cam.image
        return img

    def runKmeans(self, Z, K, criteria):  # Run Kmeans respectively for python 2/3
        if sys.version_info[0] == 3:
            return cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        else:
            return cv2.kmeans(Z, K, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    def getZValue(self, img):
        """Nobody knows what this function does because it was copy-pasted from part of the k-means example code"""
        return img.reshape((-1, 3)).astype("float32")

    def calibrateKmeans(self, img, profile, save=False, **kwargs):
        """Builds training data for the road-classifying SVM"""
        img = img[img.shape[0]//3:, :, :]  # crop

        # Initialize hyperparamaters
        K = kwargs.get("K", 5)  # How many groups for k-means to cluster into
        debug = kwargs.get("debug", False)  # Verbose/debug output enabled?
        blurSize = kwargs.get("blurSize", (5, 5))  # Gaussian blur size
        # k-means training data will be created from stepSize^(-2) pixels
        stepSize = kwargs.get("stepSize", 5)

        # Get width/height
        h, w = img.shape[:2]

        # Preprocess image
        img = cv2.GaussianBlur(img, blurSize, 0)

        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)  # Kmeans criteria

        Z = self.getZValue(img)
        ret, labels, center = self.runKmeans(Z, K, criteria)

        chsv = np.array([colorsys.rgb_to_hsv(*(c[::-1]/255.0))
                         for c in center])  # Center colors as HSV
        print("chsv", chsv)
        print("Z", Z)
        for name in profile:
            print("Processing", name)
            color_rgb = profile[name]
            print(color_rgb)
            color = np.array(colorsys.rgb_to_hsv(
                *(np.array(color_rgb)/255.0)))  # Profile color as HSV
            print(color)
            losses = np.abs(chsv-color).mean(axis=1)  # Color diffs
            print(np.mean(losses))
            n = np.argmin(losses)  # Find closest center color to profile color
            print(n)
            self.kProfile[name] = chsv[n]
            self.kLabels[n] = name
            self.kNames[name] = n
            self.kProfRGB[name] = profile[name]

        center = np.uint8(center)

        res = center[labels.flatten()]
        res2 = res.reshape((img.shape))

        labels = labels.reshape((img.shape[0], img.shape[1]))

        trainX = []
        trainY = []

        for x in range(0, labels.shape[1], stepSize):
            for y in range(0, labels.shape[0], stepSize):
                label = labels[y, x]
                if not label in self.kLabels:
                    continue
                trainX.append(img[y, x])
                trainY.append(label)

        trainX, trainY = np.array(trainX), np.array(trainY)
        print(self.kLabels)
        print("Training size:", trainX.size)
        self.clf = svm.LinearSVC()
        self.clf.fit(trainX, trainY)

        self.calibrated = True
        if save:
            self.saveSvm("../test/model.pkl")
        if debug:
            return res2

    # depracated: non functional
    def process1(color):
        def getLineColor(img, m, b, step=2):
            bottom = min(max((-b)/m, 0), img.shape[1]-1)

            top = max(min((img.shape[0]-b)/m, img.shape[1]-1), 0)

            if top < bottom:
                bottom, top = top, bottom
            total = np.array([0, 0, 0])
            size = 0

            for x in range(int(bottom), int(top), step):
                y = min(max(int(m*x+b), 0), img.shape[0]-1)
                color = img[y, x]
                total += color
                size += 1
            if size == 0:
                return None
            avg = total/size

            return avg

        height = color.shape[0]
        width = color.shape[1]
        horizonOffset = -100
        HhorizonOffset = 100
        region_of_interest_vertices = [
            (0, height),
            (width//2 - HhorizonOffset, height//2+horizonOffset),
            (width//2 + HhorizonOffset, height//2+horizonOffset),
            (width, height),
        ]
        cropped = region_of_interest(color, np.array(
            [region_of_interest_vertices], np.int32))
        img = grayscale(cropped)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        edges = autoCanny(img)

        output = color

        lines = cv2.HoughLines(edges, 1, np.pi/180, 250)

        if lines is None:
            print("no lines found")
            return output

        for line in lines:
            for rho, theta in line[:10]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))

                m = unzero((y2-y1)/(unzero(x2-x1)))
                b = y1-m*x1
                lineColor = getLineColor(color, m, b)
                if lineColor is None:
                    continue

                cv2.line(output, (0, int(b)),
                         (1000, int(m*1000+b)), tuple(lineColor), 3)
                # print(lineColor)
                cv2.circle(output, (int(x0), int(y0)), 4, (255, 0, 0), -1)

        return output

    def process3(self, imgin):
        imgin = unwarp(imgin)  # gets rid of perspective effect
        shape = imgin.shape
        pixels = shape[0]*shape[1]
        clipping = getDefault(imgin.shape[0], imgin.shape[1])
        Cimgs = []
        print(self.kNames)
        # unwarp->mask->grayscale->gaussblur->canny->houghLines
        debugOut = imgin
        for currColor in self.kNames:

            debugOut = imgin

            # svm classification:
            bools = (self.clf.predict(imgin.reshape(pixels, 3)).reshape(
                (shape[0], shape[1], 1)) == self.kNames[currColor]).astype("float")

            boolimg = bools.astype("uint8")*255

            Cimgs.append(grayscale(np.bitwise_and(imgin, boolimg)))  # masking
            # TODO: use boolimg as a mask on normal img then threshold the img to get rid of noise

            Cimg = Cimgs[-1]
            Cimg[Cimg < 180] = 0

            img = cv2.GaussianBlur(Cimg, (5, 5), 0)

            edges = autoCanny(boolimg)

            # detect lines
            lines = cv2.HoughLines(edges, 1, np.pi/180, 175)

            if lines is None:
                print("no lines found")
                return debugOut

            for line in lines:
                for rho, theta in line[:10]:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a*rho
                    y0 = b*rho
                    x1 = int(x0 + 1000*(-b))
                    y1 = int(y0 + 1000*(a))
                    x2 = int(x0 - 1000*(-b))
                    y2 = int(y0 - 1000*(a))

                    m = unzero((y2-y1)/(unzero(x2-x1)))
                    b = y1-m*x1
                    lineColor = self.kProfRGB[currColor]
                    # TODO: throw out horizontal lines

                    # for debugging, not actually nec
                    cv2.line(debugOut, (0, int(b)),
                             (1000, int(m*1000+b)), tuple(lineColor), 10)
                    cv2.circle(debugOut, (int(x0), int(y0)),
                               4, (255, 0, 0), -1)

        return debugOut

    # process4 helpers

    def getBools(self, img, colorId):
        shape = img.shape
        pixels = shape[0]*shape[1]
        return (self.clf.predict(img.reshape(pixels, 3)).reshape(
                (shape[0], shape[1], 1)) == self.kNames[colorId]).astype("float")

    def findLine(self, img, colorId, **kwargs):
        """Find the position of the base of a line along the x-axis"""
        shape = img.shape
        pixels = shape[0]*shape[1]
        bottom = shape[0]
        # A 1D array where each element is its X value
        rowMap = np.array([x for x in range(shape[1])])
        # How many pixels up from the bottom to sample
        depth = kwargs.get("cascadeDepth", 100)
        # Whether to find the mean or median of the lane pixels to find the lane marker center
        calcType = kwargs.get("center", "median")
        # Whether to denoise
        doDenoising = kwargs.get("denoise", False)
        # The extracted map of colorId colored pixels
        bools = self.getBools(img, colorId)

        if doDenoising:
            bools = denoise(img, bools) #converts to grayscale

        if calcType == "mean":
            # The sum of the X coordinates of each white pixel
            posSum = np.zeros((shape[1],))
            pixels = 0
            for y in range(bottom-depth, bottom):
                row = bools[y].reshape((shape[1],))
                posSum += rowMap*row
                # Compute how many row pixels are actually being added to the average
                pixels += (row == 1).sum()

            return posSum.sum()/pixels

        elif calcType == "median":
            posSamples = []

            for y in range(bottom-depth, bottom):
                row = bools[y].reshape((shape[1],))
                for x, cell in enumerate(row):
                    if cell == 1:
                        posSamples.append(rowMap[x])
            return np.median(np.array(posSamples))

        elif calcType == "min":
            posSum = np.zeros((shape[1],))

    def process4(self, img, debug=False, imgOut=False):
        # Position of respective lines on the X-axis
        img = img[img.shape[0]//3:, :, :]
        roadCenter = self.findLine(img, "yellow", cascadeDepth=200)
        roadEdge = self.findLine(img, "white", cascadeDepth=100)
        robotPos = img.shape[1]/2
        if debug:
            print("-----")
            print("Stats:\n")
            print("Road Center: "+str(roadCenter))
            print("Road Edge:   "+str(roadEdge))
            print("Robot Pos:   "+str(robotPos))
            print("Lane center: "+str((roadCenter+roadEdge)/2))
            print("-----")
            if imgOut:
                try:
                    drawVertical(img, int((roadCenter+roadEdge)/2),
                                 (0, 0, 255))  # BGR
                except:
                    pass
                try:
                    drawVertical(img, int(roadCenter), (255, 0, 0))
                except:
                    pass
                try:
                    drawVertical(img, int(roadEdge), (255, 0, 0))
                except:
                    pass
                return ((roadEdge, roadCenter, robotPos), img)
        return (roadEdge, roadCenter, robotPos)

    def rollingAverage(self, p4out):
        """ Smooths the values returned by process4"""
        if self.stacks == None:
            self.stacks = [[] for v in p4out]
        avgs = []
        for i, v in enumerate(p4out):
            stack = self.stacks[i]

            if v != np.nan:
                stack.append(v)

            if len(stack) > self.RAlookback:
                del stack[0]

            avg = np.array(stack).mean()
            avgs.append(avg)

        return avgs

    # svm io
    def loadSvm(self, path):
        with open(path, 'rb') as fid:
            temp = pickle.load(fid)
            self.clf = temp[0]
            self.kNames = temp[1]
            self.kLabels = temp[2]
            self.kProfRGB = temp[3]

    def saveSvm(self, path):
        with open(path, 'wb') as fid:
            pickle.dump(
                [self.clf, self.kNames, self.kLabels, self.kProfRGB], fid)


if __name__ == "__main__":
    cam = Camera(mirror=True)
    LD = LaneDetector()
    calibImg = LD.getCalibImage(cam)
    cv2.imwrite("calib.png", calibImg)
    res = LD.calibrateKmeans(calibImg, ColorProfile.lanes, debug=True)
    LD.saveSvm("model.pkl")
    while True:
        cv2.imshow('calibration img', res)
        if cv2.waitKey(1) == 27:
            break  # esc to quit

    while True:
        cv2.imshow('my webcam', LD.processr(cam.image, True, True)[0])
        if cv2.waitKey(1) == 27:
            break  # esc to quit
