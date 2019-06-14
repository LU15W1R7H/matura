from utils import *
import tensorflow as tf
import numpy as np

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28

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

print(imgs[0])
# imgs = loadGreyscaleImageDirectory('../data/trainingSample/0/')
# plotImages(img)


learning_rate = 0.001

input_layer = tf.placeholder(tf.float32, (None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), name='inputs')
target_layer = tf.placeholder(tf.float32, (None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), name='targets')

""" Encoder """
conv0 = tf.layers.conv2d(inputs=input_layer,filters=120,kernel_size=(3,3),padding='same',activation=tf.nn.relu)
maxpool0 = tf.layers.max_pooling2d(inputs=conv0,pool_size=(2,2),strides=(2,2),padding='same')

conv1 = tf.layers.conv2d(inputs=maxpool0,filters=160,kernel_size=(3,3),padding='same',activation=tf.nn.relu)
maxpool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=(2,2),strides=(2,2),padding='same')

conv2 = tf.layers.conv2d(inputs=maxpool1,filters=200,kernel_size=(3,3),padding='same',activation=tf.nn.relu)
maxpool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size=(2,2),strides=(2,2),padding='same')

conv3 = tf.layers.conv2d(inputs=maxpool2,filters=240,kernel_size=(3,3),padding='same',activation=tf.nn.relu)
maxpool3 = tf.layers.max_pooling2d(inputs=conv3,pool_size=(2,2),strides=(2,2),padding='same')

encoded = maxpool3
""" Decoder """
upsample4 = tf.layers.resize_images(inputs=encoded,size=(8,8),method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
conv4 = tf.layers.conv2d(inputs=upsample4,filters=200,kernel_size=(3,3),padding='same',activation=tf.nn.relu)

upsample5 = tf.layers.resize_images(inputs=conv4,size=(16,16),method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
conv5 = tf.layers.conv2d(inputs=upsample4,filters=160,kernel_size=(3,3),padding='same',activation=tf.nn.relu)

upsample6 = tf.layers.resize_images(inputs=conv5,size=(32,32),method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
conv6 = tf.layers.conv2d(inputs=upsample4,filters=120,kernel_size=(3,3),padding='same',activation=tf.nn.relu)

upsample7 = tf.layers.resize_images(inputs=conv6,size=(64,64),method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
conv7 = tf.layers.conv2d(inputs=upsample4,filters=15,kernel_size=(3,3),padding='same',activation=tf.nn.relu)


logits = tf.layers.conv2d(inputs=conv7,filters=3,kernel_size=(3,3),padding='same',activation=None)

decoded = tf.nn.sigmoid(logits)

loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=target_layer, logits=logits)

cost = tf.reduce_mean(loss)
opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)

sess - tf.Session()
saver = tf.train.Saver()
sess.run(tf.global_variables_initalizer())
saver.restore(sess, "")

for i, img in enumerate(imgs):
    imageio = readImage(img)
    imageio = imageio[:,:,0:3].reshape((1,64,64,3)) / 255.
    file_name = os.path.basename(img)
    denseRep = sess.run([encoded], feed_dict={inputs_layer: imageio, target_layer: imageio})
