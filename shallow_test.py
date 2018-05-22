from keras import optimizers
from keras.applications.vgg16 import VGG16, WEIGHTS_PATH_NO_TOP
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Input, Conv2D, MaxPooling2D
from keras.models import Model
import keras.preprocessing.image as im
from keras.utils.data_utils import get_file
import keras.utils.np_utils as npu
import numpy as np
import util


input_tensor = Input(shape=(250, 250, 3))

# Start of VGG16 definition
# Definition found at: https://github.com/keras-team/keras/blob/master/keras/applications/vgg16.py
# Block 1
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_tensor)
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(5, activation='softmax')(x)

model = Model(input_tensor, x, name='shallow_model')
model.compile(
    loss = 'categorical_crossentropy', 
    optimizer = optimizers.SGD(lr = 0.001, momentum = 0.9),
    metrics=['accuracy'])

# Load data from files
data1 = util.load('dataset/lfw_batch_1.dat')
data2 = util.load('dataset/lfw_batch_2.dat')
data3 = util.load('dataset/lfw_batch_3.dat')
data4 = util.load('dataset/lfw_batch_4.dat')
val_data = util.load('dataset/lfw_batch_5.dat')
# Setup input structure for training and validation
cls_1 = []
cls_2 = []
cls_3 = []
cls_4 = []
for t in (data1['labels'] + data2['labels'] + data3['labels'] + data4['labels']):
    cls_1.append(t[0] - 1)
    cls_2.append(t[1] - 1)
    cls_3.append(t[2] - 1)
    cls_4.append(t[3] - 1)

val_cls_1, val_cls_2, val_cls_3, val_cls_4 = [], [], [], []
for t in val_data['labels']:
    val_cls_1.append(t[0] - 1)
    val_cls_2.append(t[1] - 1)
    val_cls_3.append(t[2] - 1)
    val_cls_4.append(t[3] - 1)

#x_train = , data3['data'], data4['data'])) / 255
x_train = np.concatenate((data1['data'], data2['data']))
del data1['data'] 
del data2['data'] 
x_train = np.concatenate((x_train, data3['data']))
del data3['data'] 
x_train = np.concatenate((x_train, data4['data']))
del data4['data'] 
x_train = x_train / 255; 
x_val = val_data['data'] / 255
y_train = npu.to_categorical(cls_4, 5)

y_val = npu.to_categorical(val_cls_4, 5)

datagen = im.ImageDataGenerator()

batch_size = 128
updates_per_epoch = len(x_train) / batch_size
epochs = 50

print(x_val.shape)

model.fit_generator(datagen.flow(x_train, y_train, batch_size = 128), 
        steps_per_epoch = updates_per_epoch,
        epochs = epochs,
        validation_steps = int(len(x_val) / batch_size),
        validation_data = datagen.flow(x_val, y_val, batch_size = 128))
