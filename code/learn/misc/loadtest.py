import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def loadImage(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.resize(img, (512,512),interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # switch BGR encoding to RGB
    return img

def formatImg(img):
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            for c in range(3):
                img[x,y,c] /= 255. 

def writeImage(img, path):
    cv2.imwrite(path,img)

def showImage(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def plotImages(imgs):
    n=10
    plt.figure(figsize=(20,4))
    for i in range(len(imgs)):
        ax = plt.subplot(2, n, i+1)
        plt.imshow(imgs[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

def transformImage(img):
    return img.reshape(16,16)

testimg = loadImage('../../data/sunset.jpg')
showImage(testimg)
print(testimg)
arr = np.array(testimg)
print(arr)
formatImg(testimg)



