import cv2
import dlib
import os
import numpy as np
import math

########################## CONSTANTS ###################################
PATH_BASE = 'D:/trabajo/proyectos/data'      # select a valid path
PREDICTOR_FILE = os.path.join(PATH_BASE, 'shape_predictor_68_face_landmarks.dat')
########################## CONSTANTS ###################################

########################## VARIABLES ###################################
windowName = 'Window'
camera = 0
width = 600
height = 600

print('OpenCV version:', cv2.__version__)
########################## VARIABLES ###################################

########################## FUNCTIONS ###################################
# Compute similarity transform given two sets of two points.
# OpenCV requires 3 pairs of corresponding points.
# We are faking the third one.
def similarityTransform(inPoints, outPoints):
  s60 = math.sin(60*math.pi/180)
  c60 = math.cos(60*math.pi/180)

  inPts = np.copy(inPoints).tolist()
  outPts = np.copy(outPoints).tolist()

  # The third point is calculated so that the three points make an equilateral triangle
  xin = c60*(inPts[0][0] - inPts[1][0]) - s60*(inPts[0][1] - inPts[1][1]) + inPts[1][0]
  yin = s60*(inPts[0][0] - inPts[1][0]) + c60*(inPts[0][1] - inPts[1][1]) + inPts[1][1]

  inPts.append([np.int32(xin), np.int32(yin)])

  xout = c60*(outPts[0][0] - outPts[1][0]) - s60*(outPts[0][1] - outPts[1][1]) + outPts[1][0]
  yout = s60*(outPts[0][0] - outPts[1][0]) + c60*(outPts[0][1] - outPts[1][1]) + outPts[1][1]

  outPts.append([np.int32(xout), np.int32(yout)])

  # Now we can use estimateRigidTransform for calculating the similarity transform.
  tform = cv2.estimateAffinePartial2D(np.array([inPts]), np.array([outPts]))
  return tform[0]

def normalizeImageLandmark(outSize, imgIn, pointsIn):
    h, w = outSize

    # Corners of the eye in input image
    if len(pointsIn) == 68:
        eyecornerSrc = [pointsIn[36], pointsIn[45]]
    elif len(pointsIn) == 5:
        eyecornerSrc = [pointsIn[2], pointsIn[0]]

    # Corners of the eye in normalized image based on opencv information
    eyecornerDst = [(np.int32(0.3 * w), np.int32(h / 3)),
                    (np.int32(0.7 * w), np.int32(h / 3))]

    # Calculate similarity transform
    tform = similarityTransform(eyecornerSrc, eyecornerDst)
    imgOut = np.zeros(imgIn.shape, dtype=imgIn.dtype)

    # Apply similarity transform to input image
    imgOut = cv2.warpAffine(imgIn, tform, (w, h))
    return imgOut

def getPointsFromLandmarks(landmarks):
    points = []
    for p in landmarks.parts():
        pt = (p.x, p.y)
        points.append(pt)
    return points
########################## FUNCTIONS ###################################

########################## PROCESS IMAGE ###############################
# Instance face detector
faceDetector = dlib.get_frontal_face_detector()

# Instance landmark detector
landmarkDetector = dlib.shape_predictor(PREDICTOR_FILE)

# Read video
#video = cv2.VideoCapture(camera)
frame = cv2.imread(os.path.join(PATH_BASE, 'face5.jpg'))
#frame = cv2.resize(frame, (width, height))
while True:
    # Read frame
    #ok, frame = video.read()
    #if not ok:
    #    break

    faceRects = faceDetector(frame, 0)
    print('Number of faces detected:', len(faceRects))

    for i in range(0, len(faceRects)):
        # Get rectangle
        newRect = dlib.rectangle(int(faceRects[i].left()),
                                 int(faceRects[i].top()),
                                 int(faceRects[i].right()),
                                 int(faceRects[i].bottom()))

        # For every face rectangle, run landmarkDetector
        landmarks = landmarkDetector(frame, newRect)

        if len(landmarks.parts()) > 0:
            points = getPointsFromLandmarks(landmarks)
            imgOut = normalizeImageLandmark((frame.shape[0], frame.shape[1]), frame, points)
            print('Number of landmarks:', len(landmarks.parts()))

    cv2.imshow(windowName, frame)
    cv2.imshow('window2', imgOut)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
########################## PROCESS IMAGE ###############################


