import cv2
import dlib
import os
import numpy as np
import math
import sys

########################## CONSTANTS ###################################
PATH_BASE = 'D:/trabajo/proyectos/data'      # select a valid path
PREDICTOR_FILE = os.path.join(PATH_BASE, 'shape_predictor_68_face_landmarks.dat')
########################## CONSTANTS ###################################

########################## VARIABLES ###################################
windowName = 'Window'
camera = 0
width = 600
height = 600
isFirstFrame = True
eyeDistanceNotCalculated = True
dotRadius = 0
showStabilized = True

# Initializing the parameters
points=[]
pointsPrev=[]
pointsDetectedCur=[]
pointsDetectedPrev=[]

print('OpenCV version:', cv2.__version__)
########################## VARIABLES ###################################

########################## FUNCTIONS ###################################
def getPointsFromLandmarks(landmarks):
    points = []
    for p in landmarks.parts():
        pt = (p.x, p.y)
        points.append(pt)
    return points

def getIntereyeDistance(points):
    leftEyeLeftCorner = (points[36][0], points[36][1])
    rightEyeRightCorner = (points[45][0], points[45][1])
    distance = cv2.norm(np.array(rightEyeRightCorner) - np.array(leftEyeLeftCorner))
    distance = int(distance)
    return distance
########################## FUNCTIONS ###################################

########################## PROCESS IMAGE ###############################
# Instance face detector
faceDetector = dlib.get_frontal_face_detector()

# Instance landmark detector
landmarkDetector = dlib.shape_predictor(PREDICTOR_FILE)

video = cv2.VideoCapture(camera)

# Read first frame
ok, framePrev = video.read()
if not ok:
    print('There is no first frame')
    sys.exit()

frameGrayPrev = cv2.cvtColor(framePrev, cv2.COLOR_BGR2GRAY)

while True:
    # Read frame
    ok, frame = video.read()
    if not ok:
        break

    frameDlib = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faceRects = faceDetector(frame, 0)
    print('Number of faces detected:', len(faceRects))

    if len(faceRects) == 0:
        print('No face detected')
    else:
        for i in range(0, len(faceRects)):
            # Get rectangle
            newRect = dlib.rectangle(int(faceRects[i].left()),
                                     int(faceRects[i].top()),
                                     int(faceRects[i].right()),
                                     int(faceRects[i].bottom()))

            # For every face rectangle, run landmarkDetector
            landmarks = landmarkDetector(frameDlib, newRect)

            # Handling the first frame of video differently,for the first frame copy the current frame points
            if isFirstFrame:
                pointsPrev = getPointsFromLandmarks(landmarks)
                pointsDetectedPrev = getPointsFromLandmarks(landmarks)

            # If not the first frame, copy points from previous frame
            else:
                pointsPrev = points
                pointsDetectedPrev = pointsDetectedCur

            # pointsDetectedCur stores results returned by the facial landmark detector
            # points stores the stabilized landmark points
            points = getPointsFromLandmarks(landmarks)
            pointsDetectedCur = getPointsFromLandmarks(landmarks)

            # Convert to numpy float array
            pointsArr = np.array(points, np.float32)
            pointsPrevArr = np.array(pointsPrev, np.float32)

            # If eye distance is not calculated before
            if eyeDistanceNotCalculated:
                eyeDistance = getIntereyeDistance(points)
                print(eyeDistance)
                eyeDistanceNotCalculated = False

            if eyeDistance > 100:
                dotRadius = 3
            else:
                dotRadius = 2

            sigma = eyeDistance * eyeDistance / 400
            s = 2 * int(eyeDistance / 4) + 1

            #  Set up optical flow params
            lk_params = dict(winSize=(s, s), maxLevel=5, criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 0.03))

            pointsArr, status, err = cv2.calcOpticalFlowPyrLK(frameGrayPrev, frameGray, pointsPrevArr, pointsArr, **lk_params)

            # Converting to float
            pointsArrFloat = np.array(pointsArr, np.float32)

            # Converting back to list
            points = pointsArrFloat.tolist()

            # Final landmark points are a weighted average of
            # detected landmarks and tracked landmarks
            for k in range(0, len(landmarks.parts())):
                d = cv2.norm(np.array(pointsDetectedPrev[k]) - np.array(pointsDetectedCur[k]))
                alpha = math.exp(-d*d/sigma)
                points[k] = (1 - alpha) * np.array(pointsDetectedCur[k]) + alpha * np.array(points[k])

            # Drawing over the stabilized landmark points
            if showStabilized is True:
                for p in points:
                    cv2.circle(frame, (int(p[0]), int(p[1])), dotRadius, (255, 0, 0), -1)
            else:
                for p in pointsDetectedCur:
                    cv2.circle(frame, (int(p[0]), int(p[1])), dotRadius, (0, 0, 255), -1)

            isFirstFrame = False

    cv2.imshow(windowName, frame)

    k = cv2.waitKey(1) & 0xFF

    # Use spacebar to toggle between Stabilized and Unstabilized version.
    if k == 32:
        showStabilized = not showStabilized

    if k == 27:
        break

    framePrev = frame
    frameGrayPrev = frameGray
########################## PROCESS IMAGE ###############################

