from os.path import isfile, join
import json
import matplotlib.pyplot as plt
import os

data_dir = 'train_data'

resnet18 = {}
resnet34 = {}
resnet50 = {}

with open('train_data/scratch_extended_resnet18', 'r') as file:
    resnet18 = json.load(file)

with open('train_data/scratch_extended_resnet34', 'r') as file:
    resnet34 = json.load(file)

with open('train_data/extended_resnet50', 'r') as file:
    resnet50 = json.load(file)

x_vals = list(range(1, 51))
p1 = plt.subplot()
p1.plot(x_vals, resnet18['val_loss'], 'red', label='resnet18 validation loss', linestyle='-')
p1.plot(x_vals, resnet18['loss'], 'red', label='resnet18 training loss', linestyle='--', linewidth=0.5)

p1.plot(x_vals, resnet34['val_loss'], 'blue', label='resnet34 validation loss', linestyle='-')
p1.plot(x_vals, resnet34['loss'], 'blue', label='resnet34 training loss', linestyle='-', linewidth=0.5)

p1.plot(x_vals, resnet50['val_loss'], 'green', label='resnet50 validation loss', linestyle='-')
p1.plot(x_vals, resnet50['loss'], 'green', label='resnet50 training loss', linestyle='-', linewidth=0.5)
p1.legend(loc='best')

plt.figure()
p2 = plt.subplot()
p2.plot(x_vals, resnet18['val_acc'], 'red', label='resnet18 validation acc', linestyle='-')
p2.plot(x_vals, resnet18['acc'], 'red', label='resnet18 training acc', linestyle='--', linewidth=0.5)

p2.plot(x_vals, resnet34['val_acc'], 'blue', label='resnet34 validation acc', linestyle='-')
p2.plot(x_vals, resnet34['acc'], 'blue', label='resnet34 training acc', linestyle='-', linewidth=0.5)

p2.plot(x_vals, resnet50['val_acc'], 'green', label='resnet50 validation acc', linestyle='-')
p2.plot(x_vals, resnet50['acc'], 'green', label='resnet50 training acc', linestyle='-', linewidth=0.5)
p2.legend(loc='best')

plt.show()
