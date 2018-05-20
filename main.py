import sys
sys.path.append('/usr/local/lib/python3.6/dist-packages')

import keras.datasets.cifar10 as cf10
import keras.preprocessing.image as im
import keras.utils.np_utils as npu
import numpy as np
import os
import os.path
import tensorflow as tf
import util
import cmd

# Silence debug-info (relevant for tensorflow-gpu)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'

args = cmd.validate_cmdargs(sys.argv)

# Consider splitting some training data into validation data
(x_train, y_train), (x_test, y_test) = cf10.load_data()

#x_train = x_train[0 : 64, :, :, :]
#y_train = y_train[0 : 64, :]

# Training Parameters
NUM_CLASSES = 10
batch_size = 32
updates_per_epoch = len(x_train) / batch_size
epochs = 50

# Turn labels into one-hot encodings
y_train = npu.to_categorical(y_train, NUM_CLASSES)
y_test = npu.to_categorical(y_test, NUM_CLASSES)

# Fetch the specified model (or create it if there is none)
model = util.get_model(args.model)

# Data batch generator
datagen = im.ImageDataGenerator()

# Tensorflow with GPU enabled easily runs out of memory. 
# So we need to train and evaluate seperately
if args.mode == 'train':

    # Computes necessary details for feature normalizations (flip/rotate/shift/etc)
    datagen.fit(x_train)

    # Test-data datagen validator
    testdatagen = im.ImageDataGenerator()

    res = model.fit_generator(
        datagen.flow(x_train, y_train, batch_size),
        validation_data = testdatagen.flow(x_test, y_test, batch_size),
        steps_per_epoch = updates_per_epoch,
        epochs = epochs)
    util.save_model(model, args.model)
    
    if args.save_train_data:
        util.save_train_data(model.name, res.history)
    if args.plot:
        util.plot(res.history)
elif args.mode == 'evaluate':
    accuracy = []
    for i in range(100):
        x_batch = x_test[i*10:(i+1)*10]
        y_batch = y_test[i*10:(i+1)*10]
        res = model.test_on_batch(x_batch, y_batch)
        accuracy.append(res[1])
    print("Average accuracy:", sum(accuracy) / len(accuracy))
