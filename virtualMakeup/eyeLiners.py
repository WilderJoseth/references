import cv2
import dlib
import numpy as np
import os
from scipy.interpolate import interp1d
import sys

########################## CONSTANTS ###################################
PATH_BASE = 'D:\\trabajo\\proyectos\\data'      # select a valid path
PREDICTOR_FILE = os.path.join(PATH_BASE, 'shape_predictor_68_face_landmarks.dat')
PATH_VIDEO = os.path.join(PATH_BASE, 'smileDetectionOutput.avi')
########################## CONSTANTS ###################################

########################## VARIABLES ###################################
camera = 0
windowName = 'Window'
width = 600
height = 600

print('OpenCV version:', cv2.__version__)
########################## VARIABLES ###################################

########################## FUNCTIONS ###################################
def getPoints(landmarks):
    '''

    :param landmarks:
    :return:
    '''

    points = []
    for l in range(len(landmarks.parts())):
        x, y = landmarks.part(l).x, landmarks.part(l).y
        points.append((x, y))
    return points

def getPointsValuesFunction(points):
    x = points[:, 0]
    y = points[:, 1]

    xi = np.arange(int(x.min()), int(x.max()))
    f = interp1d(x, y, kind='quadratic')
    yi = f(xi)

    pts = np.array((xi, yi)).T.astype(int)
    return pts

def getEyePoints(points):
    eyeLeftTop = points[36:40]
    eyeLeftTop = np.array(eyeLeftTop)
    eyeLeftTop[0][0] -= 5
    eyeLeftTop[3][0] += 5
    eyeLeftTop = getPointsValuesFunction(eyeLeftTop)

    eyeLeftBottom = points[39:42] + [points[36]]
    eyeLeftBottom = np.array(eyeLeftBottom)
    eyeLeftBottom[3][0] -= 5
    eyeLeftBottom[0][0] += 5
    eyeLeftBottom = getPointsValuesFunction(eyeLeftBottom)

    eyeRightTop = points[42:46]
    eyeRightTop = np.array(eyeRightTop)
    eyeRightTop[0][0] -= 5
    eyeRightTop[3][0] += 5
    eyeRightTop = getPointsValuesFunction(eyeRightTop)

    eyeRightBottom = points[45:48] + [points[42]]
    eyeRightBottom = np.array(eyeRightBottom)
    eyeRightBottom[3][0] -= 5
    eyeRightBottom[0][0] += 5
    eyeRightBottom = getPointsValuesFunction(eyeRightBottom)

    return eyeLeftTop, eyeRightTop, eyeLeftBottom, eyeRightBottom

def drawLine(image, points, thickness = 4):
    image = cv2.polylines(image, [points], False, (0, 0, 0), thickness)
    return image

def getEmpty(*args):
    pass
########################## FUNCTIONS ###################################

########################## PROCESS IMAGE ###################################
# Instance face detector
faceDetector = dlib.get_frontal_face_detector()

# The landmark detector is implemented in the shape_predictor class
# Instance landmark detector
landmarkDetector = dlib.shape_predictor(PREDICTOR_FILE)

# Source image
#img = cv2.imread(os.path.join(PATH_BASE, 'smiling-man.jpg'))
img = cv2.imread(os.path.join(PATH_BASE, 'girl-no-makeup.jpg'))
#img = cv2.imread(os.path.join(PATH_BASE, 'trump.jpg'))
img = cv2.resize(img, (width, height))

faceRects = faceDetector(img, 0)
print('Number of faces detected:', len(faceRects))

cv2.namedWindow(windowName)
cv2.createTrackbar('Apply', windowName, 0, 1, getEmpty)
cv2.createTrackbar('Thickness', windowName, 2, 6, getEmpty)
while True:
    imgCopy = img.copy()
    for i in range(0, len(faceRects)):
        # Get rectangle
        newRect = dlib.rectangle(int(faceRects[i].left()),
                                 int(faceRects[i].top()),
                                 int(faceRects[i].right()),
                                 int(faceRects[i].bottom()))

        # For every face rectangle, run landmarkDetector
        landmarks = landmarkDetector(imgCopy, newRect)

        # Print number of landmarks
        if i == 0:
            print('Number of landmarks:', len(landmarks.parts()))

        if len(landmarks.parts()) > 0:
            points = getPoints(landmarks)

            e = cv2.getTrackbarPos('Apply', windowName)
            if e == 1:
                pointsTopLeft, pointsTopRight, pointsBottomLeft, pointsBottomRight = getEyePoints(points)

                t = cv2.getTrackbarPos('Thickness', windowName)
                drawLine(imgCopy, pointsTopLeft, t)
                drawLine(imgCopy, pointsTopRight, t)
                drawLine(imgCopy, pointsBottomLeft, t)
                drawLine(imgCopy, pointsBottomRight, t)

    cv2.imshow(windowName, imgCopy)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
########################## PROCESS IMAGE ###################################

########################## REFERENCES ###################################
# https://www.data-stats.com/applying-eyelashes-and-lipstick-using-opencv/
# https://www.programmersought.com/article/38627055837/
# https://www.youtube.com/watch?v=JPraJGpGDG8
########################## REFERENCES ###################################
