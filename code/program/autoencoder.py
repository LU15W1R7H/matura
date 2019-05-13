import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard

# load data

# define model
bottleneck = 32

input_data = Input(shape=(64,64,1)) #choose image size
p = Conv2D(16, (3,3), activation='relu',padding='same')(input_data)
p = MaxPooling2D((2,2),padding='same')(x)
p = Conv2D(8,(3,3),activation='relu',padding='same')(p)
p = MaxPooling2D((2,2),padding='same')(x)
p = Conv2D(8,(3,3),activation='relu',padding='same')(p)
encoded = MaxPooling2D((2,2),padding='same')(x)

p = Conv2D(8,(3,3),activation='relu',padding='same')(encoded)
p = UpSampling2D((2,2))(p)
p = Conv2D(8,(3,3),activation='relu',padding='same')(p)
p = UpSampling2D((2,2))(p)
p = Conv2D(16,(3,3),activation='relu',padding='same')(p)
p = UpSampling2D((2,2))(p)
decoded = Conv2D(1,(3,3),activation='sigmoid',padding='same')(p)

autoencoder = Model(input_data,decoded)
autoencoder.compile(optimizer='adadelta',loss='binary_crossentropy')

autoencoder.fit(x_train,x_train,epochs=50,batch_size=128,shuffle=True,validation_data=(x_test,_x_test),callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

decoded_faces = autoencoder.predict(x_test)


autoencoder.save('faceautoencoder.model')
