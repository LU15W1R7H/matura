import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

# hyperparameters
BATCH_SIZE = 128
EPOCHS = 100

# load MNIST dataset
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

# format data
x_train = x_train.astype('float32') / 255.
x_test = np.reshape(x_test, (len(x_test),28,28,1))
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train),28,28,1))

# generate noise images / inputs
noise_factor = 0.8
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# define model
input_data = tf.keras.Input(shape=(28,28,1))
econv0 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1),padding='same',activation='relu')(input_data)
emaxpool0 = tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=None,padding='same')(econv0)
econv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1),padding='same',activation='relu')(emaxpool0)
emaxpool1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=None,padding='same')(econv1)
encoded =  emaxpool1 # bottleneck (shape=(7,7,32))
dconv0 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu')(encoded)
dupsample0 = tf.keras.layers.UpSampling2D(size=(2,2), interpolation='nearest')(dconv0)
dconv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu')(dupsample0)
dupsample1 = tf.keras.layers.UpSampling2D(size=(2,2), interpolation='nearest')(dconv1)
dconv2 = tf.keras.layers.Conv2D(filters=1,kernel_size=(3,3), strides=(1,1), padding='same', activation='sigmoid')(dupsample1)
decoded = dconv2

# compile model
autoencoder = tf.keras.Model(input_data, decoded)
autoencoder.compile(optimizer='sgd',loss='mean_squared_error')

print(autoencoder.summary())

# train model and save it
autoencoder.fit(x=x_train_noisy, y=x_train, batch_size=128, epochs=100, shuffle=True,validation_data=(x_test_noisy,x_test),callbacks=[tf.keras.callbacks.TensorBoard(log_dir='/tmp/denoiser',histogram_freq=0,write_graph=False)])
autoencoder.save('denoiser.model')

# let model make predictions
decoded_imgs = autoencoder.predict(x_test)

n = 10 # jeweils 10 Bilder
plt.figure()
for i in range(n):
    # verrauschte Bilder
    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(x_test_noisy[i].reshape(28,28))
    plt.gray()

    # entrauschte Bilder
    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()

    # Original-Bilder
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()

plt.show()
