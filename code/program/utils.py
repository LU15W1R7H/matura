import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import itertools

# group iterable in groups and return list of groups
def grouper(n, iterable):
    args = [iter(iterable)] * n
    return ([e for e in t if e != None] for t in itertools.zip_longest(*args))

# loads image by path, boolean greyscale
def loadImage(path,greyscale):
    if greyscale:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # switch BGR encoding to RGB

    if img is None:
        return None

    img = np.asarray(img)
    return img

# returns resized image
def resizeImage(img, newWidth, newHeight):
    img = cv2.resize(img, (newWidth, newHeight), interpolation=cv2.INTER_CUBIC)
    return img

# plots list of images
def plotImages(imgs):
    cols=10
    rows = len(imgs)/cols + 1
    fig = plt.figure(figsize=(8,8))
    for i in range(len(imgs)):
        ax = fig.add_subplot(rows, cols, i+1)
        plt.imshow(imgs[i])
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
    plt.show()

# loads images by list of paths
def loadImages(image_paths, greyscale):
    imgs=[]
    for path in image_paths:
        img = loadImage(path,greyscale)
        if(img is not None):
            imgs.append(img)
    return np.asarray(imgs)

def getFilePaths(dirPath):
    paths = []
    for filename in os.listdir(dirPath):
        paths.append(os.path.join(dirPath,filename))

    return paths

def getImagePathsInDir(dirPath):
    paths = []
    for filepath in glob.glob(path+ '*.jpg'):
        paths.append(filepath)

    return paths


def writeImage(img, path):
    cv2.imwrite(path,img)
