import cv2
import dlib
import numpy as np
import os

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

def getEyePoints(landmarks):
    positionLeft = [37, 38, 40, 41]
    positionRight = [43, 44, 46, 47]

    pointsLeft = []
    pointsRight = []
    for i in positionLeft:
        pointsLeft.append((landmarks.part(i).x, landmarks.part(i).y))

    for i in positionRight:
        pointsRight.append((landmarks.part(i).x, landmarks.part(i).y))

    ML = cv2.moments(np.array(pointsLeft))
    cXL = int(ML["m10"] / ML["m00"])
    cYL = int(ML["m01"] / ML["m00"])
    ptcL = (cXL, cYL)
    rL = int((((landmarks.part(37).x - landmarks.part(40).x) ** 2 + (landmarks.part(37).y - landmarks.part(40).y) ** 2) ** (0.5)) / 3)

    MR = cv2.moments(np.array(pointsRight))
    cXR = int(MR["m10"] / MR["m00"])
    cYR = int(MR["m01"] / MR["m00"])
    ptcR = (cXR, cYR)
    rR = int((((landmarks.part(43).x - landmarks.part(46).x) ** 2 + (landmarks.part(43).y - landmarks.part(46).y) ** 2) ** (0.5)) / 3)

    return ptcL, ptcR, rL, rR

def detectEyes(image, points):
    eyeDistance = ((points[39][0] - points[42][0]) ** 2 + (points[39][1] - points[42][1]) ** 2) ** (0.5)
    minEyeDistance = int(eyeDistance / 3)
    eyeRadius = ((points[37][0] - points[38][0]) ** 2 + (points[37][1] - points[38][1]) ** 2) ** (0.5)
    minEyeRadius = int(eyeRadius / 4)
    maxEyeRadius = int(eyeRadius / 2)
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageBlur = cv2.medianBlur(imageGray, 5)

    circles = cv2.HoughCircles(imageBlur, cv2.HOUGH_GRADIENT, 1, minEyeDistance, 450, 10, minEyeRadius, maxEyeRadius)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)

def createBox(image, points, scale = 3):
    x, y, w, h = cv2.boundingRect(points)
    imageCrop = image[y:y+h, x:x+w]
    imageCrop = cv2.resize(imageCrop, (0, 0), None, scale, scale)
    return imageCrop

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
img = cv2.imread(os.path.join(PATH_BASE, 'face.jpg'))
#img = cv2.imread(os.path.join(PATH_BASE, 'trump.jpg'))
img = cv2.resize(img, (width, height))

faceRects = faceDetector(img, 0)
print('Number of faces detected:', len(faceRects))

cv2.namedWindow(windowName)
cv2.createTrackbar('Red', windowName, 0, 255, getEmpty)
cv2.createTrackbar('Blue', windowName, 0, 255, getEmpty)
cv2.createTrackbar('Green', windowName, 0, 255, getEmpty)

while True:
    imgCopy = img.copy()
    imgCrop = None
    imgMask = np.zeros_like(imgCopy)
    imgColor = np.zeros_like(imgMask)

    b = cv2.getTrackbarPos('Blue', windowName)
    r = cv2.getTrackbarPos('Red', windowName)
    g = cv2.getTrackbarPos('Green', windowName)
    imgColor[:] = b, g, r

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
        points = getPoints(landmarks)
        points = np.array(points)
        #imgCrop = createBox(imgCopy, points[36:47])

        #ptcL, ptcR, rL, rR = getEyePoints(landmarks)
        #cv2.circle(imgMask, center=ptcR, radius=rR, color=(255, 255, 255), thickness=-1)
        #cv2.circle(imgMask, center=ptcL, radius=rL, color=(255, 255, 255), thickness=-1)

        #imgColor = cv2.bitwise_and(imgMask, imgColor)
        #imgColor = cv2.GaussianBlur(imgColor, (7, 7), 10)
        #imgColor = cv2.addWeighted(imgCopy, 1, imgColor, 0.4, 0)

        detectEyes(imgCopy, points)

    cv2.imshow(windowName, imgCopy)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
########################## PROCESS IMAGE ###################################

########################## REFERENCES ###################################
# https://www.youtube.com/watch?v=lhIwiQMIoYo
########################## REFERENCES ###################################

