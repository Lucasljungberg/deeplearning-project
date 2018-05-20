from PIL import Image
import numpy as np
import pickle

def dump (obj, fname):
    with open(fname, 'wb') as file:
        pickle.dump(obj, file)

def load (fname):
    with open(fname, 'rb') as file:
        data = pickle.load(file)
    return data

data = load('dataset/lfw_batch_1.dat')
print(data['labels'])
# labels = load('dataset/lfw_classes.dat')
# for i in range(1, 6):
#     content = {}
#     fname = 'dataset/lfw_batch_%d.dat' %i
#     data = load(fname)
#     content['data'] = data
#     content['labels'] = labels[i*1000:i*1000 + data.shape[0]]
#     dump(content, fname)

# classes = []
# images = []
# imfiles = []
# with open('dataset/training.txt', 'r') as f:
#     for line in f:
#         tokens = line.split(' ')
#         imfiles.append(tokens[0])
#         cls1 = int(tokens[-4])
#         cls2 = int(tokens[-3])
#         cls3 = int(tokens[-2])
#         cls4 = int(tokens[-1])
#         classes.append((cls1, cls2, cls3, cls4))
# print(len(imfiles))
# dump(classes, 'dataset/lfw_classes.dat')
# del classes
# classes = None
# part = 1
# for imf, i in zip(imfiles, range(len(imfiles))):
#     if (i % 1000) == 0 and i is not 0:
#         images = np.array(images)
#         print(images.shape)
#         images = images.reshape((1000, 250, 250, 3))
#         dump(images, 'dataset/lfw_batch_%d.dat' %part)
#         del images
#         images = []
#         part += 1
#     if 'lfw' not in imf:
#         break
#     with Image.open('dataset/' + imf) as im:
#         images.append(list(im.getdata()))
# images = np.array(images).reshape((i % 1000, 250, 250, 3))
# dump(images, 'dataset/lfw_batch_%d.dat' %part)
