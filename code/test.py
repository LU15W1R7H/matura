import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
# verrauschte Bilder (Input) generieren
noise_factor = 1

# für x_train
noise_matrices = np.random.normal(loc=0.0, scale=1.0, size=(64,64))
noise_matrices *= noise_factor

plt.figure()
ax = plt.subplot(1, 1, 1)
plt.imshow(noise_matrices)
plt.gray()
plt.show()
