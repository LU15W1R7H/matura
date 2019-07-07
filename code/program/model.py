from utils import *
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from facealigner import FaceAligner
import time

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
IMAGE_CHANNELS = 1
LEARNING_RATE = 0.002


# MODELL
input_data = tf.keras.Input(shape=(IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_CHANNELS)) # shape=(IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_CHANNELS)
# encoder
econv0 = tf.keras.layers.Conv2D(filters=120,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu',use_bias=True)(input_data) # input_shape=(IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_CHANNELS),
maxpool0 = tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=None,padding='same')(econv0)
econv1 = tf.keras.layers.Conv2D(filters=160,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu',use_bias=True)(maxpool0)
maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=None,padding='same')(econv1)
econv2 = tf.keras.layers.Conv2D(filters=200,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu',use_bias=True)(maxpool1)
maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=None,padding='same')(econv2)

# bottleneck
encoded = maxpool2

#decoder
dconv0 = tf.keras.layers.Conv2D(filters=200,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu',use_bias=True)(encoded)
upsample0 = tf.keras.layers.UpSampling2D(size=(2,2),interpolation='nearest')(dconv0)
dconv1 = tf.keras.layers.Conv2D(filters=160,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu',use_bias=True)(upsample0)
upsample1 = tf.keras.layers.UpSampling2D(size=(2,2),interpolation='nearest')(dconv1)
dconv2 = tf.keras.layers.Conv2D(filters=120,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu',use_bias=True)(upsample1) # padding is valid!!!
upsample2 = tf.keras.layers.UpSampling2D(size=(2,2),interpolation='nearest')(dconv2)
dconv3 = tf.keras.layers.Conv2D(filters=1,kernel_size=(3,3),strides=(1,1),padding='same',activation='sigmoid')(upsample2)

decoded = dconv3

autoencoder = tf.keras.Model(input_data,decoded)
autoencoder.compile(optimizer='adadelta',loss='binary_crossentropy')
print(autoencoder.summary())

# loading data
aligner = FaceAligner()

dirPaths = getFilePaths('/home/luis/ml_data/')
dirPaths = dirPaths[0:500] # max 50'000
paths = [getFilePaths(path) for path in dirPaths]
paths = np.concatenate(paths).ravel()

imgs = loadImages(paths, greyscale=True)
for i in range(len(imgs)):
    faces = aligner.getFaces(imgs[i])
    if(len(faces)==0):
        continue
    landmarks = aligner.getLandmarks(imgs[i],faces[0])
    imgs[i] = aligner.align(imgs[i],landmarks)

np.random.shuffle(imgs)
imgs = np.asarray([resizeImage(img, IMAGE_WIDTH,IMAGE_HEIGHT) for img in imgs])
# for img in imgs:
#     cv2.imshow('test',img)
#     cv2.waitKey(0)
imgs = imgs.astype('float32') / 255.0 # normalize data
imgs = imgs[...,np.newaxis]

# training

autoencoder.fit(x=imgs,y=imgs,batch_size=128,epochs=25,shuffle=True,validation_data=None,callbacks=[tf.keras.callbacks.TensorBoard(log_dir='/tmp/autoencoder')])

generated_face = autoencoder.predict(imgs)

autoencoder.save('myautoencoder.model')
