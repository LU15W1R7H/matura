import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

# zu testende Hyperparameter
EPOCHS = 20

BATCH_SIZES = [ 16, 32, 64, 128, 256 ]
FILTER_NUMS = [ 16, 32, 64, 128, 256 ]
ACTIVATIONFUNCTIONS = [ 'sigmoid', 'relu' ]

# MNIST-Datensatz laden
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

# Daten formatieren
x_train = x_train.astype('float32') / 255.0
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
x_test = x_test.astype('float32') / 255.0
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))

# verrauschte Bilder (Input) generieren
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0.0, 1.0)
x_test_noisy = np.clip(x_test_noisy, 0.0, 1.0)

# verschiedene Modelle testen
for activation_function in ACTIVATIONFUNCTIONS:
    for batch_size in BATCH_SIZES:
        for num_filters in FILTER_NUMS:

            input_data = tf.keras.Input(shape=(28,28,1))
            econv0 = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(3,3), strides=(1,1),padding='same',activation=activation_function, kernel_initializer='glorot_normal')(input_data)
            emaxpool0 = tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=None,padding='same')(econv0)
            econv1 = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(3,3), strides=(1,1),padding='same',activation=activation_function, kernel_initializer='glorot_normal')(emaxpool0)
            emaxpool1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=None,padding='same')(econv1)
            encoded =  emaxpool1 # bottleneck (shape=(7,7,32))
            dconv0 = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(3,3), strides=(1,1), padding='same', activation=activation_function, kernel_initializer='glorot_normal')(encoded)
            dupsample0 = tf.keras.layers.UpSampling2D(size=(2,2), interpolation='nearest')(dconv0)
            dconv1 = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(3,3), strides=(1,1), padding='same', activation=activation_function, kernel_initializer='glorot_normal')(dupsample0)
            dupsample1 = tf.keras.layers.UpSampling2D(size=(2,2), interpolation='nearest')(dconv1)
            dconv2 = tf.keras.layers.Conv2D(filters=1,kernel_size=(3,3), strides=(1,1), padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(dupsample1)
            decoded = dconv2

            # Modell kompilieren
            autoencoder = tf.keras.Model(input_data, decoded)
            autoencoder.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])

            print(autoencoder.summary())

            # Training und TensorBoard
            NAME = 'denoiser-{}-{}-batches-{}-filters'.format(activation_function, batch_size, num_filters)
            tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs/{}'.format(NAME))
            autoencoder.fit(x=x_train_noisy, y=x_train, batch_size=batch_size, epochs=EPOCHS, shuffle=True, validation_data=(x_test_noisy,x_test), callbacks=[tensorboard])
            autoencoder.save('models/{}', NAME)
