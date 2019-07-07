from utils import *
import numpy as np
import cv2
import dlib
from collections import OrderedDict

class FaceAligner:


    def __init__(self):

        self.FACIAL_LANDMARKS_68_DICT = OrderedDict([
            ('mouth', (48, 68)),
            ('inner_mouth', (60, 68)),
            ('right_eyebrow', (17, 22)),
            ('left_eyebrow', (22, 27)),
            ('right_eye', (36, 42)),
            ('left_eye', (42, 48)),
            ('nose', (27, 36)),
            ('jaw', (0, 17))
        ])

        self.detector = dlib.get_frontal_face_detector() # to detect the faces
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') # to detect landmarks

        self.faceSize=256
        self.leftEyePos=(0.35,0.35)
        self.rightEyePos=(1.0-self.leftEyePos[0],1.0-self.leftEyePos[1])

    def getFaces(self,img):
        rects = self.detector(img)
        return rects

    def faceToBox(self,face):
        x = face.left()
        y = face.top()
        w = face.right() - x
        h = face.bottom() - y
        return (x,y,w,h)

    def getLandmarks(self,img,face):
        landmarks = self.predictor(img,face)
        coords = np.zeros((68,2),dtype=int)
        for i in range(0,68):
            coords[i] = (landmarks.part(i).x,landmarks.part(i).y)
        return coords

    def drawBB(self,img,face):
        (x,y,w,h) = face_to_bb(face)
        cv2.rectangle(gray, (x,y), (x+w, y+h), (0,255,0), 2) # draw bounding box

    def drawLandmarks(self,img,landmarks):
        for (x,y) in landmarks:
            cv2.circle(img, (x,y),1, (0,0,255),-1)

    def align(self,img,landmarks):

        # get eye landmarks
        (lStart,lEnd) = self.FACIAL_LANDMARKS_68_DICT['left_eye']
        (rStart,rEnd) = self.FACIAL_LANDMARKS_68_DICT['right_eye']
        leftEyePts = landmarks[lStart:lEnd]
        rightEyePts = landmarks[rStart:rEnd]

        # compute center of mass for eyes
        leftEyeCtr = leftEyePts.mean(axis=0).astype('int')
        rightEyeCtr = rightEyePts.mean(axis=0).astype('int')

        # compute angle between eye centroids
        dX = rightEyeCtr[0] - leftEyeCtr[0]
        dY = rightEyeCtr[1] - leftEyeCtr[1]
        angle = np.degrees(np.arctan2(dY,dX)) - 180

        # determine scale
        dS = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (self.rightEyePos[0] - self.leftEyePos[0])
        desiredDist *= self.faceSize
        scale = desiredDist / dS

        # get center between eyes
        eyesCenter = ((leftEyeCtr[0] + rightEyeCtr[0]) // 2, (leftEyeCtr[1] + rightEyeCtr[1]) // 2) # // is a integer division

        # create matrix for transformation
        mat = cv2.getRotationMatrix2D(eyesCenter,angle,scale)

        # update translation component of matrix
        tX = self.faceSize * 0.5
        tY = self.faceSize * self.leftEyePos[1]
        mat[0,2] += (tX - eyesCenter[0])
        mat[1,2] += (tY - eyesCenter[1])

        # apply transformation
        output = cv2.warpAffine(img,mat,(self.faceSize,self.faceSize),flags=cv2.INTER_CUBIC)

        return output
