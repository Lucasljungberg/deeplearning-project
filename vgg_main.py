import cmd
import keras.preprocessing.image as im
import keras.utils.np_utils as npu
import numpy as np
import resnet101_mtfl
import sys
import util
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from keras.utils import np_utils

args = cmd.validate_cmdargs(sys.argv)

batch_size = 128
nb_classes = 10
epochs = 50
data_augmentation = True


# input image dimensions
img_rows, img_cols = 32, 32
# The CIFAR10 images are RGB.
img_channels = 3

# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Convert class vectors to binary class matrices.
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# subtract mean and normalize
mean_image = np.mean(x_train, axis=0)
x_train -= mean_image
x_test -= mean_image
x_train /= 128.
x_test /= 128.

updates_per_epoch = len(x_train) / batch_size


model = util.get_model(args.model)
datagen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True)

datagen.fit(x_train)

if args.mode == 'train':
    res = model.fit_generator(
       # util.multiclass_gen(x_train, (y_train1, y_train2, y_train3, y_train4), batch_size),
        datagen.flow(x_train, y_train, batch_size = 128), 
        steps_per_epoch = updates_per_epoch,
        epochs = epochs,
        validation_steps = int(len(x_test) / batch_size),
        validation_data = datagen.flow(x_test, y_test, batch_size = 128))
      #  validation_data = util.multiclass_gen(x_val, (y_val1, y_val2, y_val3, y_val4), batch_size))

    util.save_model(model, args.model)

    if args.save_train_data:
        util.save_train_data(model.name, res.history)
    if args.plot:
        util.plot(res.history)
