import cv2
import dlib
import numpy as np
import os
import sys

########################## CONSTANTS ###################################
PATH_BASE = 'data'      # select a valid path
PREDICTOR_FILE = os.path.join(PATH_BASE, 'shape_predictor_68_face_landmarks.dat')
########################## CONSTANTS ###################################

########################## VARIABLES ###################################
windowName = 'Window'
width = 600
height = 600

print('OpenCV version:', cv2.__version__)
########################## VARIABLES ###################################

########################## FUNCTIONS ###################################
def getPoints(landmarks):
    '''
    Convert landmarks to list of points
    :param landmarks:
    :return:
    '''

    points = []
    for l in range(len(landmarks.parts())):
        x, y = landmarks.part(l).x, landmarks.part(l).y
        points.append((x, y))
    return points

def drawMask(image, points):
    '''
    Draw a polygon
    :param image:
    :param points:
    :return:
    '''
    imageMask = np.zeros_like(image)
    imageMask = cv2.fillPoly(imageMask, [points], (255, 255, 255))
    return imageMask

def getEmpty(*args):
    '''
    Empty function for trackbar
    :param args:
    :return:
    '''
    pass
########################## FUNCTIONS ###################################

########################## PROCESS IMAGE ###################################
# Instance face detector
faceDetector = dlib.get_frontal_face_detector()

# The landmark detector is implemented in the shape_predictor class
# Instance landmark detector
landmarkDetector = dlib.shape_predictor(PREDICTOR_FILE)

# Source image
img = cv2.imread(os.path.join(PATH_BASE, 'girl-no-makeup.jpg'))

# Resize for visualization
img = cv2.resize(img, (width, height))

faceRects = faceDetector(img, 0)
print('Number of faces detected:', len(faceRects))

# Create trackbars
cv2.namedWindow(windowName)
cv2.createTrackbar('Red', windowName, 0, 255, getEmpty)
cv2.createTrackbar('Blue', windowName, 0, 255, getEmpty)
cv2.createTrackbar('Green', windowName, 0, 255, getEmpty)

while True:
    imgCopy = img.copy()
    imgMask = None

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
            points = np.array(points)

            # Create mask of lips
            imgMask = drawMask(imgCopy, points[48:67])
            imgColor = np.zeros_like(imgMask)

            # Get colors
            b = cv2.getTrackbarPos('Blue', windowName)
            r = cv2.getTrackbarPos('Red', windowName)
            g = cv2.getTrackbarPos('Green', windowName)

            # Set colors
            imgColor[:] = b, g, r

            # Adapt the color to maks
            imgColor = cv2.bitwise_and(imgMask, imgColor)
            imgColor = cv2.GaussianBlur(imgColor, (7, 7), 10)

            # Assing color to image
            imgColor = cv2.addWeighted(imgCopy, 1, imgColor, 0.6, 0)

    cv2.imshow(windowName, imgColor)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
########################## PROCESS IMAGE ###################################

########################## REFERENCES ###################################
# https://www.youtube.com/watch?v=V2gmgkSqyi8
########################## REFERENCES ###################################
