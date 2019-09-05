import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = np.reshape(x_test, (len(x_test),28,28,1))
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train),28,28,1))

noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

autoencoder = tf.keras.models.load_model('denoiser.model') # already compiled

print(autoencoder.summary())

decoded_imgs = autoencoder.predict(x_test)

n = 10 # jeweils 10 Bilder
plt.figure()
for i in range(n):
    # verrauschte Bilder
    ax = plt.subplot(3, n, i+1)
    plt.imshow(x_test_noisy[i].reshape(28,28))
    plt.gray()

    # entrauschte Bilder
    ax = plt.subplot(3, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()

    # Original-Bilder
    ax = plt.subplot(3, n, i+1+2*n)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()

plt.show()
