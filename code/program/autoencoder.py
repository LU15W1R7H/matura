import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard

print("Test")

def machine_learning():

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

def other_way():
    # basis
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(150,150,30)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy',optimizer='rmsprop')

    model.fit(epochs=100)

    # convolutional and pooling approach
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150,150,3)), # 3 bytes for pixels
        #maybe change input_shape because throwing away borders for convolutions (no neighbors) 148x148
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        #flatten for feeding into DNN
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        #512 neuron hidden layer
       tf.keras.layers.Dense(512,activation='relu'),
        #output
        tf.keras.layers.Dense(3,activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy',optimizer='rmsprop')

    model.fit()

    model.save('path/to/model')

    new_model = tf.keras.models.load_model('same/path')
    new_model.summary()
