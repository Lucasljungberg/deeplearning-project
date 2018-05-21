from keras import optimizers
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Model, load_model
import json
import matplotlib.pyplot as plt
import numpy as np
import os.path
import pickle
import vgg16
import vgg19
from resnet.resnet101 import Scale
import resnet101_mtfl as resnet101
import resnet152_mtfl as resnet152
import resnet50 as resnet50

paths = {
    'vgg16': "models/extended_vgg16",
    'vgg19': "models/extended_vgg19",
    'resnet101': "models/extended_resnet101",
    'resnet152': "models/extended_resnet152"
    'resnet18': "models/extended_resnet18",
    'resnet50': "models/extended_resnet50"
}
models = {
    'vgg16': vgg16,
    'vgg19': vgg19,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'resnet50': resnet50
}

def save_model(model, name):
    model.save(paths.get(name, 'models/temporary_model'))
    print("Saved model successfully!")

def get_model (name):
    if (os.path.isfile(paths.get(name))):
        print("Found existing model")
        return load_model(paths.get(name), custom_objects={'Scale': Scale})
    else:
        print("Could not find existing '%s' model... creating extended model from scratch" %name)
        return models[name].create_model()

def save_train_data (name, data):
    with open('train_data/' + name, 'w') as f:
        f.write(json.dumps(data))

def plot (history):
    x_axis = [i for i in range(len(history['loss']))]
    plt.figure()

    plt.title('Loss')
    p1 = plt.subplot()
    p1.plot(x_axis, history['loss'], 'blue', label='Training loss')
    p1.plot(x_axis, history['val_loss'], 'red', label='Validation loss')
    p1.legend(loc="best")

    plt.figure()

    plt.title('Accuracy')
    p2 = plt.subplot()
    p2.plot(x_axis, history['acc'], 'blue', label='Training accuracy')
    p2.plot(x_axis, history['val_acc'], 'red', label='Validation accuracy')
    p2.legend(loc="best")

    plt.show()

def dump (obj, fname):
    with open(fname, 'wb') as file:
        pickle.dump(obj, file)

def load (fname):
    with open(fname, 'rb') as file:
        data = pickle.load(file)
    return data

def multiclass_gen (x, outputs, bs):
    """
    Specifically designed for 4 outputs.
    :param: x The input pool
    :outputs: A list of 4 one-hot label pools
    :bs: Batch size
    """
    pool_size = len(x)
    idx = 0
    while True:
        idxs = range(idx, min(pool_size, idx + bs))
        x_batch = x[idxs]
        y0_batch = outputs[0][idxs]
        y1_batch = outputs[1][idxs]
        y2_batch = outputs[2][idxs]
        y3_batch = outputs[3][idxs]
        yield x_batch, [y0_batch, y1_batch, y2_batch, y3_batch]
