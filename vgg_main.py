import cmd
import keras.preprocessing.image as im
import keras.utils.np_utils as npu
import numpy as np
import resnet101_mtfl
import sys
import util
from keras.preprocessing.image import ImageDataGenerator

args = cmd.validate_cmdargs(sys.argv)

batch_size = 32
nb_classes = 10
nb_epoch = 200
data_augmentation = True


# input image dimensions
img_rows, img_cols = 32, 32
# The CIFAR10 images are RGB.
img_channels = 3

# The data, shuffled and split between train and test sets:
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Convert class vectors to binary class matrices.
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# subtract mean and normalize
mean_image = np.mean(X_train, axis=0)
X_train -= mean_image
X_test -= mean_image
X_train /= 128.
X_test /= 128.

updates_per_epoch len(x_train) / batch_size


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
        datagen.flow(x_train, y_train1, batch_size = 128), 
        steps_per_epoch = updates_per_epoch,
        epochs = epochs,
        validation_steps = int(len(x_val) / batch_size),
        validation_data = datagen.flow(x_val, y_val1, batch_size = 128))
      #  validation_data = util.multiclass_gen(x_val, (y_val1, y_val2, y_val3, y_val4), batch_size))

    util.save_model(model, args.model)

    if args.save_train_data:
        util.save_train_data(model.name, res.history)
    if args.plot:
        util.plot(res.history)
