from utils import *
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
IMAGE_CHANNELS = 1

dirPaths = getFilePaths('../data/trainingSample/')
paths = [getFilePaths(path) for path in dirPaths]
paths = np.asarray(paths)
paths = paths.ravel()

imgs = loadImages(paths, greyscale=True)

print(imgs[0])

imgs = imgs.astype(float)

for img in imgs:
    for x in range(len(img)):
        for y in range(len(img[x])):
            img[x,y] /= 255.0

print(imgs[0].shape)
# imgs = loadGreyscaleImageDirectory('../data/trainingSample/0/')
# plotImages(img)

input_data = tf.keras.Input(shape=(IMAGE_WIDTH,IMAGE_HEIGHT)) #shape=(IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH)
econv0 = tf.keras.layers.Conv2D(filters=16,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu')(input_data)
maxpool0 = tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=None,padding='same')(econv0)
econv1 = tf.keras.layers.Conv2D(filters=8,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu')(maxpool0)
maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=None,padding='same')(econv1)
econv2 = tf.keras.layers.Conv2D(filters=8,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu')(maxpool1)
maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=None,padding='same')(econv2)

encoded = maxpool2

dconv0 = tf.keras.layers.Conv2D(filters=8,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu')(encoded)
upsample0 = tf.keras.layers.UpSampling2D(size=(2,2),interpolation='nearest')(dconv0)
dconv1 = tf.keras.layers.Conv2D(filters=8,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu')(upsample0)
upsample1 = tf.keras.layers.UpSampling2D(size=(2,2),interpolation='nearest')(dconv1)
dconv2 = tf.keras.layers.Conv2D(filters=16,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu')(upsample1)
upsample2 = tf.keras.layers.UpSampling2D(size=(2,2),interpolation='nearest')(dconv2)

decoded = upsample2


autoencoder = tf.keras.Model(input_data,decoded)
autoencoder.compile(optimizer='adadelta',loss='binary_crossentropy')

autoencoder.fit(x=imgs,y=imgs,batch_size=128,epochs=50,shuffle=True,validation_data=None,callbacks=[tf.keras.callbacks.TensorBoard(log_dir='/tmp/autoencoder')])

generated_face = autoencoder.predict(imgs[0])

autoencoder.save('myautoencoder.model')


# -----------------------------


# for i, img in enumerate(imgs):
#     imageio = readImage(img)
#     imageio = imageio[:,:,0:3].reshape((1,64,64,3)) / 255.
#     file_name = os.path.basename(img)
#     denseRep = sess.run([encoded], feed_dict={inputs_layer: imageio, target_layer: imageio})
