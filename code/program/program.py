from utils import *
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import time

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
IMAGE_CHANNELS = 1

dirPaths = getFilePaths('../data/trainingSample/')
paths = [getFilePaths(path) for path in dirPaths]
paths = np.asarray(paths)
paths = paths.ravel()

imgs = loadImages(paths, greyscale=True)
imgs = imgs.astype('float32') / 255.0 # normalize data

# MODELL
input_data = tf.keras.Input(shape=(IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_CHANNELS))
# encoder
econv0 = tf.keras.layers.Conv2D(filters=16,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu',use_bias=True)(input_data) # input_shape=(IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_CHANNELS),
maxpool0 = tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=None,padding='same')(econv0)
econv1 = tf.keras.layers.Conv2D(filters=8,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu',use_bias=True)(maxpool0)
maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=None,padding='same')(econv1)
econv2 = tf.keras.layers.Conv2D(filters=8,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu',use_bias=True)(maxpool1)
maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=None,padding='same')(econv2)

# bottleneck
encoded = maxpool2

#decoder
dconv0 = tf.keras.layers.Conv2D(filters=8,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu',use_bias=True)(encoded)
upsample0 = tf.keras.layers.UpSampling2D(size=(2,2),interpolation='nearest')(dconv0)
dconv1 = tf.keras.layers.Conv2D(filters=8,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu',use_bias=True)(upsample0)
upsample1 = tf.keras.layers.UpSampling2D(size=(2,2),interpolation='nearest')(dconv1)
dconv2 = tf.keras.layers.Conv2D(filters=16,kernel_size=(3,3),strides=(1,1),padding='valid',activation='relu',use_bias=True)(upsample1) # padding is valid!!!
upsample2 = tf.keras.layers.UpSampling2D(size=(2,2),interpolation='nearest')(dconv2)
dconv3 = tf.keras.layers.Conv2D(filters=1,kernel_size=(3,3),strides=(1,1),padding='same',activation='sigmoid')(upsample2)

decoded = dconv3

autoencoder = tf.keras.Model(input_data,decoded)
autoencoder.compile(optimizer='adadelta',loss='binary_crossentropy')

print(autoencoder.summary())

autoencoder.fit(x=imgs,y=imgs,batch_size=128,epochs=50,shuffle=True,validation_data=None,callbacks=[tf.keras.callbacks.TensorBoard(log_dir='/tmp/autoencoder')])

generated_face = autoencoder.predict(imgs)

autoencoder.save('myautoencoder.model')


# for i, img in enumerate(imgs):
#     imageio = readImage(img)
#     imageio = imageio[:,:,0:3].reshape((1,64,64,3)) / 255.
#     file_name = os.path.basename(img)
#     denseRep = sess.run([encoded], feed_dict={inputs_layer: imageio, target_layer: imageio})
