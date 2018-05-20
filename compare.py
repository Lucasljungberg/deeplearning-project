from os.path import isfile, join
import json
import matplotlib.pyplot as plt
import os

data_dir = 'train_data'

vgg16 = {}
vgg19 = {}

with open('train_data/extended_vgg16', 'r') as file:
    vgg16 = json.load(file)

with open('train_data/extended_vgg19', 'r') as file:
    vgg19 = json.load(file)

x_vals = list(range(1, 51))
p1 = plt.subplot()
p1.plot(x_vals, vgg16['val_loss'], 'red', label='vgg16 validation loss', linestyle='-')
p1.plot(x_vals, vgg16['loss'], 'red', label='vgg16 training loss', linestyle='--', linewidth=0.5)

p1.plot(x_vals, vgg19['val_loss'], 'blue', label='vgg19 validation loss', linestyle='-')
p1.plot(x_vals, vgg19['loss'], 'blue', label='vgg19 training loss', linestyle='-', linewidth=0.5)
p1.legend(loc='best')

plt.figure()
p2 = plt.subplot()
p2.plot(x_vals, vgg16['val_acc'], 'red', label='vgg16 validation acc', linestyle='-')
p2.plot(x_vals, vgg16['acc'], 'red', label='vgg16 training acc', linestyle='--', linewidth=0.5)

p2.plot(x_vals, vgg19['val_acc'], 'blue', label='vgg19 validation acc', linestyle='-')
p2.plot(x_vals, vgg19['acc'], 'blue', label='vgg19 training acc', linestyle='-', linewidth=0.5)
p2.legend(loc='best')

plt.show()
