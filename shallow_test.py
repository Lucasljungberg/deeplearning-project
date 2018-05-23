from keras import optimizers
from keras.applications.vgg19 import VGG19, WEIGHTS_PATH_NO_TOP
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Input, Conv2D, MaxPooling2D
from keras.models import Model
from keras.utils.data_utils import get_file
import keras.preprocessing.image as im
import keras.utils.np_utils as npu
import numpy as np
import util
import vgg19


input_tensor = Input(shape=(250, 250, 3))

model = vgg19.create_model(shape=(250, 250, 3), classifiers = 5)

# Load data from files
data1 = util.load('dataset/lfw_batch_1.dat')
# data2 = util.load('dataset/lfw_batch_2.dat')
# data3 = util.load('dataset/lfw_batch_3.dat')
# data4 = util.load('dataset/lfw_batch_4.dat')
val_data = util.load('dataset/lfw_batch_5.dat')
# Setup input structure for training and validation
labels = []
for t in (data1['labels']):
    labels.append(t[3] - 1)

val_labels = []
for t in val_data['labels']:
    val_labels.append(t[3] - 1)

x_train = data1['data'] / 255
x_val = val_data['data'] / 255
y_train = npu.to_categorical(labels, 5)

y_val = npu.to_categorical(val_labels, 5)

datagen = im.ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

batch_size = 32
updates_per_epoch = len(x_train) / batch_size
epochs = 50

print(x_val.shape)

model.fit_generator(datagen.flow(x_train, y_train, batch_size = batch_size), 
        steps_per_epoch = updates_per_epoch,
        epochs = epochs,
        validation_steps = int(len(x_val) / batch_size),
        validation_data = datagen.flow(x_val, y_val, batch_size = batch_size))
