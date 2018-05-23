from keras import optimizers
from keras.applications.vgg19 import WEIGHTS_PATH_NO_TOP
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Input, Conv2D, MaxPooling2D
from keras.models import Model
from keras.utils.data_utils import get_file
import util
import os.path

model_name = 'vgg19_extended_model'

def create_model(shape=(32, 32, 3), classifiers = 10, train = False):
    input_tensor = Input(shape=shape)

    # Start of VGG19 definition
    # Definition found at: https://github.com/keras-team/keras/blob/master/keras/applications/vgg19.py
    # Block 1
<<<<<<< HEAD
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', trainable=train)(input_tensor)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', trainable=train)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=train)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=train)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', trainable=train)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', trainable=train)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', trainable=train)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4', trainable=train)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', trainable=train)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', trainable=train)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', trainable=train)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4', trainable=train)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', trainable=train)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', trainable=train)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', trainable=train)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4', trainable=train)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    # End of VGG19 definition

    vgg19 = Model(input_tensor, x, name='custom_vgg19')

    vgg19_output = vgg19.output
    x = Flatten()(vgg19_output)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(classifiers, activation='softmax')(x)

    model = Model(input_tensor, x, name='extended_vgg19')
    model.compile(
        loss = 'categorical_crossentropy', 
        optimizer = optimizers.SGD(lr = 0.001, momentum = 0.9),
        metrics=['accuracy'])

    return model
