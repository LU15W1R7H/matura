import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

# hyperparameters
EPOCHS = 20

POSSIBLE_BATCH_SIZES = [ 16, 32, 64, 128, 256]
POSSIBLE_NUMS_FILTER = [ 16, 32, 64, 128, 256]
POSSIBLE_ACITVATIONFUNCTIONS = [ 'sigmoid', 'relu' ]


# load MNIST dataset
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

# format data
x_train = x_train.astype('float32') / 255.
x_test = np.reshape(x_test, (len(x_test),28,28,1))
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train),28,28,1))

# generate noise images / inputs
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# test different models
for activation_function in POSSIBLE_ACITVATIONFUNCTIONS:
    for batch_size in POSSIBLE_BATCH_SIZES:
        for num_filters in POSSIBLE_NUMS_FILTER:

            input_data = tf.keras.Input(shape=(28,28,1))
            econv0 = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(3,3), strides=(1,1),padding='same',activation=activation_function)(input_data)
            emaxpool0 = tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=None,padding='same')(econv0)
            econv1 = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(3,3), strides=(1,1),padding='same',activation=activation_function)(emaxpool0)
            emaxpool1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=None,padding='same')(econv1)
            encoded =  emaxpool1 # bottleneck (shape=(7,7,32))
            dconv0 = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(3,3), strides=(1,1), padding='same', activation=activation_function)(encoded)
            dupsample0 = tf.keras.layers.UpSampling2D(size=(2,2), interpolation='nearest')(dconv0)
            dconv1 = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(3,3), strides=(1,1), padding='same', activation=activation_function)(dupsample0)
            dupsample1 = tf.keras.layers.UpSampling2D(size=(2,2), interpolation='nearest')(dconv1)
            dconv2 = tf.keras.layers.Conv2D(filters=1,kernel_size=(3,3), strides=(1,1), padding='same', activation='sigmoid')(dupsample1)
            decoded = dconv2

            # compile model
            autoencoder = tf.keras.Model(input_data, decoded)
            autoencoder.compile(optimizer='sgd',loss='binary_crossentropy', metrics=['accuracy'])

            print(autoencoder.summary())

            # train model and save it
            NAME = '{}-{}-batch-{}-filters-{}-denoiser'.format(loss, activation_function, batch_size, num_filters)
            tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs/{}'.format(NAME))
            autoencoder.fit(x=x_train_noisy, y=x_train, batch_size=batch_size, epochs=EPOCHS, shuffle=True,validation_data=(x_test_noisy,x_test),callbacks=[tensorboard])
