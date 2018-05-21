import cmd
import keras.preprocessing.image as im
import keras.utils.np_utils as npu
import numpy as np
import resnet101_mtfl
import sys
import util

args = cmd.validate_cmdargs(sys.argv)

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

x_train = np.concatenate((data1['data'], data2['data'], data3['data'], data4['data'])) / 255
x_val = val_data['data'] / 255
y_train1 = npu.to_categorical(cls_1, 2)
y_train2 = npu.to_categorical(cls_2, 2)
y_train3 = npu.to_categorical(cls_3, 2)
y_train4 = npu.to_categorical(cls_4, 5)

y_val1 = npu.to_categorical(val_cls_1, 2)
y_val2 = npu.to_categorical(val_cls_2, 2)
y_val3 = npu.to_categorical(val_cls_3, 2)
y_val4 = npu.to_categorical(val_cls_4, 5)


batch_size = 32
updates_per_epoch = len(x_train) / batch_size
epochs = 50
print(x_train.shape)

model = util.get_model(args.model)

if args.mode == 'train':
    res = model.fit_generator(
        util.multiclass_gen(x_train, (y_train1, y_train2, y_train3, y_train4), batch_size),
        steps_per_epoch = updates_per_epoch,
        epochs = epochs,
        validation_steps = int(len(x_val) / batch_size),
        validation_data = util.multiclass_gen(x_val, (y_val1, y_val2, y_val3, y_val4), batch_size))

    util.save_model(model, args.model)

    if args.save_train_data:
        util.save_train_data(model.name, res.history)
    if args.plot:
        util.plot(res.history)
