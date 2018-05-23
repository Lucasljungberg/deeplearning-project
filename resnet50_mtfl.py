import sys
sys.path.append('/usr/local/lib/python3.6/dist-packages')

import keras.datasets.cifar10 as cf10
import keras.preprocessing.image as im
import keras.utils.np_utils as npu
from keras.applications.resnet50 import ResNet50
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, ZeroPadding2D, BatchNormalization
import numpy as np
import os
import os.path
import tensorflow as tf
from keras.models import Model, load_model
from keras import optimizers, regularizers

# Weights file path
weight_file = 'weights/resnet50_weights_tf_dim_ordering_tf_kernels.h5'

extended_resnet_model_path = "models/extended_resnet50"
# Silence debug-info (relevant for tensorflow-gpu)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'

# Parameters
class_outputs = [2, 2, 2, 5]
output_names = ['mf', 'glasses', 'smile', 'tilt']

def create_model ():
    # Create pretrained ResNet50 model
    input_tensor = Input(shape=(250, 250, 3))
    resnet_model = ResNet50(weights='imagenet', include_top=False, input_tensor = input_tensor)
    # Freeze the model - We don't want to train it
    for layer in resnet_model.layers:
        layer.trainable = False

    # Extend the model for transfer learning
    outputs = []
    for i in range(1):
        ext_model = resnet_model.output
        ext_model = Flatten()(ext_model)
        #ext_model = Dense(1024, activation='relu')(ext_model)
        #ext_model = Dropout(0.5)(ext_model)
        ext_model = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01))(ext_model)
        outputs.append(Dense(2, activation='softmax', name = output_names[i])(ext_model))

    # Create and compile the new extended model
    model = Model(input_tensor, outputs[0], name = 'resnet50')
    model.compile(
        loss = 'categorical_crossentropy', 
        optimizer = optimizers.SGD(lr = 0.001, momentum = 0.9),
        metrics=['accuracy'])

    return model




