import sys
sys.path.append('/usr/local/lib/python3.6/dist-packages')

import keras.datasets.cifar10 as cf10
import keras.preprocessing.image as im
import keras.utils.np_utils as npu
from keras.applications.resnet50 import ResNet50
from keras.layers import Input, Dense, Activation, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, ZeroPadding2D, BatchNormalization
import numpy as np
import os
import os.path
import tensorflow as tf
import resnet_keras
from keras.models import Model, load_model
from keras import optimizers


# Weights file path
weight_file = 'weights/resnet50_weights_tf_dim_ordering_tf_kernels.h5'

extended_resnet_model_path = "models/extended_resnet50"
# Silence debug-info (relevant for tensorflow-gpu)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'

# Exit early if no mode is given
if len(sys.argv) < 2:
    print('A mode needs to be specified')
    print('Usage: python main.py [train|evaluate]')
    exit()

# Consider splitting some training data into validation data
(x_train, y_train), (x_test, y_test) = cf10.load_data()

# Parameters
NUM_CLASSES = 10
batch_size = 32
updates_per_epoch = len(x_train) / batch_size
epochs = 1

# Add padding to the inner parts of the input to fit the 200x200 minimum requirement
# Using 'mean' strategy
x_train = np.pad(x_train, [(0, 0), (84, 84), (84, 84), (0, 0)], 'mean')
x_test = np.pad(x_test, [(0, 0), (84, 84), (84, 84), (0, 0)], 'mean')

# Turn labels into one-hot encodings
y_train = npu.to_categorical(y_train, NUM_CLASSES)
y_test = npu.to_categorical(y_test, NUM_CLASSES)

# Fetch the model (or create it if there is none)
model = get_model()

# Data batch generator
datagen = im.ImageDataGenerator()

# Silence debug-info (relevant for tensorflow-gpu)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'

# Exit early if no mode is given
if len(sys.argv) < 2:
    print('A mode needs to be specified')
    print('Usage: python main.py [train|evaluate]')
    exit()

# Tensorflow with GPU enabled easily runs out of memory. 
# So we need to train and evaluate seperately
mode = os.sys.argv[1]
if mode == 'train':

    # Computes necessary details for feature normalizations (flip/rotate/shift/etc)
    datagen.fit(x_train)

    model.fit_generator(
        datagen.flow(x_train, y_train, batch_size),
        steps_per_epoch = updates_per_epoch,
        epochs = epochs)
    save_model(model)
elif mode == 'evaluate':
    for i in range(100):
        x_batch = x_test[i*10:(i+1)*10]
        y_batch = y_test[i*10:(i+1)*10]
        res = model.test_on_batch(x_batch, y_batch)
        print(res)


def save_model(model):
    model.save(extended_resnet_model_path)
    print("Saved model successfully!")

def get_model ():
    if (os.path.isfile(extended_resnet_model_path)):
        print("Found existing model")
        return load_model(extended_resnet_model_path)
    else:
        print("Could not find existing model... creating extended model from scratch")
        return create_extended_model()

def create_extended_model ():
    # Create pretrained ResNet50 model
    resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(200, 200, 3))
    # Freeze the model - We don't want to train it
    for layer in resnet_model.layers:
        layer.trainable = False

    # Extend the model for transfer learning
    ext_model = resnet_model.output
    ext_model = Flatten()(ext_model)
    ext_model = Dense(1024, activation='relu')(ext_model)
    ext_model = Dropout(0.5)(ext_model)
    ext_model = Dense(1024, activation='relu')(ext_model)
    output = Dense(10, activation='softmax')(ext_model)

    # Create and compile the new extended model
    model = Model(inputs = resnet_model.input, outputs = output)
    model.compile(
        loss = 'categorical_crossentropy', 
        optimizer = optimizers.SGD(lr = 0.001, momentum = 0.9),
        metrics=['accuracy'])

    return model
