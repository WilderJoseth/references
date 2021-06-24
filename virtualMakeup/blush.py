import cv2
import dlib
import numpy as np
import os

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
def getCenterPointsTriangle(landmarks):
    '''
    Get position of the cheeks
    :param landmarks:
    :return:
    '''

    # Select points close to left cheek
    ptL1 = (landmarks.part(36).x, landmarks.part(36).y)
    ptL2 = (landmarks.part(3).x, landmarks.part(3).y)
    ptL3 = (landmarks.part(31).x, landmarks.part(31).y)

    # Get the distance between two points in order to calculate cheek radio
    rL = int((((landmarks.part(31).x - landmarks.part(3).x)**2 + (landmarks.part(31).y - landmarks.part(3).y)**2)**(0.5))/3)

    # Get center left cheek
    ML = cv2.moments(np.array([ptL1, ptL2, ptL3]))
    cXL = int(ML["m10"] / ML["m00"])
    cYL = int(ML["m01"] / ML["m00"])
    ptcL = (cXL, cYL)

    # Select points close to right cheek
    ptR1 = (landmarks.part(45).x, landmarks.part(45).y)
    ptR2 = (landmarks.part(13).x, landmarks.part(13).y)
    ptR3 = (landmarks.part(35).x, landmarks.part(35).y)

    # Get the distance between two points in order to calculate cheek radio
    rR = int((((landmarks.part(35).x - landmarks.part(13).x) ** 2 + (landmarks.part(35).y - landmarks.part(13).y) ** 2) ** (0.5)) / 3)

    # Get center left cheek
    MR = cv2.moments(np.array([ptR1, ptR2, ptR3]))
    cXR = int(MR["m10"] / MR["m00"])
    cYR = int(MR["m01"] / MR["m00"])
    ptcR = (cXR, cYR)

    return ptcL, ptcR, rL, rR

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

        # Get cheek points
        ptcL, ptcR, rL, rR = getCenterPointsTriangle(landmarks)

        # Draw cheeks
        cv2.circle(imgMask, center=ptcR, radius=rR, color=(255, 255, 255), thickness=-1)
        cv2.circle(imgMask, center=ptcL, radius=rL, color=(255, 255, 255), thickness=-1)

        # Adapt the color to maks
        imgColor = cv2.bitwise_and(imgMask, imgColor)
        imgColor = cv2.GaussianBlur(imgColor, (7, 7), 10)

        # Assing color to image
        imgColor = cv2.addWeighted(imgCopy, 1, imgColor, 0.25, 0)

    cv2.imshow(windowName, imgColor)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
########################## PROCESS IMAGE ###################################

########################## REFERENCES ###################################
# https://www.pyimagesearch.com/2016/02/01/opencv-center-of-contour/
# https://jwdinius.github.io/blog/2020/virtualmakeup/
########################## REFERENCES ###################################
