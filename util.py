from keras import optimizers
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Model, load_model
import json
import matplotlib.pyplot as plt
import numpy as np
import os.path
import vgg16
import vgg19

paths = {
    'vgg16': "models/extended_vgg16",
    'vgg19': "models/extended_vgg19"
}
models = {
    'vgg16': vgg16,
    'vgg19': vgg19
}

def save_model(model, name):
    model.save(paths.get(name, 'models/temporary_model'))
    print("Saved model successfully!")

def get_model (name):
    if (os.path.isfile(paths.get(name))):
        print("Found existing model")
        return load_model(paths.get(name))
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
