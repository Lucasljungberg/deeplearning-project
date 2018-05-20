from keras import optimizers
from keras.applications.vgg16 import VGG16, WEIGHTS_PATH_NO_TOP
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Input, Conv2D, MaxPooling2D
from keras.models import Model
from keras.utils.data_utils import get_file
import resnet
import util

class_outputs = [2, 2, 2, 5]
output_names = ['mf', 'glasses', 'smile', 'tilt']

def create_model ():
    input_tensor = Input(shape=(250, 250, 3))
    resnet101 = resnet.ResNet101(weights = 'imagenet', 
        include_top = False,
        input_tensor = input_tensor)

    for layer in resnet101.layers:
        layer.trainable = False

    resnet_output = resnet101.output

    outputs = []
    for i in range(4):
        x = Flatten()(resnet_output)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(1024, activation='relu')(x)
        outputs.append(Dense(class_outputs[i], activation='softmax', name = output_names[i])(x))

    model = Model(input_tensor, outputs, name = 'resnet101_mtfl')
    model.compile(
        loss = 'categorical_crossentropy', 
        optimizer = optimizers.SGD(lr = 0.001, momentum = 0.9),
        metrics=['accuracy'])

    return model
