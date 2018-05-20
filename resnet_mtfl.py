from keras import optimizers
from keras.applications.vgg16 import VGG16, WEIGHTS_PATH_NO_TOP
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Input, Conv2D, MaxPooling2D
from keras.models import Model
from keras.utils.data_utils import get_file
import resnet
import util

def create_model ():
    input_tensor = Input(shape=(250, 250, 3))
    resnet101 = resnet.ResNet101(weights = 'imagenet', 
        include_top = False,
        input_tensor = input_tensor)

    for layer in resnet101.layers:
        layer.trainable = False

    resnet_output = resnet101.output

    x = Flatten()(resnet_output)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    cls1 = Dense(2, activation='softmax', name = "1")(x)
    cls2 = Dense(2, activation='softmax', name = "2")(x)
    cls3 = Dense(2, activation='softmax', name = "3")(x)
    cls4 = Dense(5, activation='softmax', name = "4")(x)

    model = Model(input_tensor, [cls1, cls2, cls3, cls4], name = 'resnet101_mtfl')
    model.compile(
        loss = 'categorical_crossentropy', 
        optimizer = optimizers.SGD(lr = 0.001, momentum = 0.9),
        metrics=['accuracy'])

    return model
