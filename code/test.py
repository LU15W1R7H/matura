import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

# MNIST-Datensatz laden
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

# Daten formatieren
x_train = x_train.astype('float32') / 255.0
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
x_test = x_test.astype('float32') / 255.0
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))

# verrauschte Bilder (Input) generieren
noise_factor = 0.5

# für x_train
noise_matrices = np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
noise_matrices *= noise_factor
x_train_noisy = x_train + noise_matrices
x_train_noisy = np.clip(x_train_noisy, 0.0, 1.0)

# für x_test in Kurzfassung
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_test_noisy = np.clip(x_test_noisy, 0.0, 1.0)
