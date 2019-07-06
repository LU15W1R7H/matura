from utils import *
import numpy as np
import cv2
import dlib
from collections import OrderedDict

# converts rectangle to bounding box
def face_to_bb(face):
    x = face.left()
    y = face.top()
    w = face.right() - x
    h = face.bottom() - y
    return (x,y,w,h)

FACIAL_LANDMARKS_68_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("inner_mouth", (60, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 36)),
	("jaw", (0, 17))
])

#For dlib’s 5-point facial landmark detector:

# DETECTOR
detector = dlib.get_frontal_face_detector()
# we always use the predictor with 68 landmarks
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def detectFaces(img):
    rects = detector(img)
    return rects

def getLandmarks(img,face):
    landmarks = predictor(img,face)
    coords = np.zeros((68,2),dtype=int)
    for i in range(0,68):
        coords[i] = (landmarks.part(i).x,landmarks.part(i).y)

    return coords

# --------------------

# ALIGNER
desiredLeftEye=(0.35,0.35)
desiredFaceSize=256

def align(img,landmarks):
    (lStart,lEnd) = FACIAL_LANDMARKS_68_IDXS['left_eye']
    (rStart,rEnd) = FACIAL_LANDMARKS_68_IDXS['right_eye']

    leftEyePts = landmarks[lStart:lEnd]
    rightEyePts = landmarks[rStart:rEnd]

    # compute center of mass for eyes
    leftEyeCenter = leftEyePts.mean(axis=0).astype('int')
    rightEyeCenter = rightEyePts.mean(axis=0).astype('int')

    # compute angle between eye centroids
    dY = rightEyeCenter[1] - leftEyeCenter[1]
    dX = rightEyeCenter[0] - leftEyeCenter[0]
    angle = np.degrees(np.arctan2(dY,dX)) - 180

    desiredRightEyeX = 1.0 - desiredLeftEye[0]

    # determine scale
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desiredDist = (desiredRightEyeX - desiredLeftEye[0])
    desiredDist *= desiredFaceSize
    scale = desiredDist / dist

    # get center between eyes
    eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2, (leftEyeCenter[1] + rightEyeCenter[1]) // 2)

    # create matrix for affine projection
    M = cv2.getRotationMatrix2D(eyesCenter,angle,scale)

    # update translation of matrix
    tX = desiredFaceSize * 0.5
    tY = desiredFaceSize * desiredLeftEye[1]
    M[0,2] += (tX - eyesCenter[0])
    M[1,2] += (tY - eyesCenter[1])

    # apply matrix
    (w,h) = (desiredFaceSize, desiredFaceSize)
    output = cv2.warpAffine(img,M,(w,h),flags=cv2.INTER_CUBIC)

    return output




# -----------------------

dirPaths = getFilePaths('/home/luis/ml_data/')
dirPaths = dirPaths[8:9]
paths = [getFilePaths(path) for path in dirPaths]
paths = np.concatenate(paths).ravel()


# paths = np.asarray(paths)
# paths = paths.ravel()

imgs = loadImages(paths, greyscale=True)
imgs = [resizeImage(img, 265,265) for img in imgs]

gray = imgs[0]
cv2.imshow('lol',gray)
cv2.waitKey(0)

faces = detectFaces(gray)

for face in faces:
    landmarks = getLandmarks(gray,face)
    (x,y,w,h) = face_to_bb(face)
    cv2.rectangle(gray, (x,y), (x+w, y+h), (0,255,0), 2) # draw bounding box

    for (x,y) in landmarks:
        cv2.circle(gray, (x,y),1, (0,0,255),-1)

    cv2.imshow('Output',gray)
    cv2.waitKey(0)

    aligned = align(gray,landmarks)
    cv2.imshow('aligned',aligned)
    cv2.waitKey(0)

# globals
