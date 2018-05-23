from resnetBuild import ResnetBuilder
from keras import optimizers

def create_model (shape = (32, 32, 3), classifiers = 10, train = False):
    model = ResnetBuilder.build_resnet_34(shape, classifiers)
    model.name = 'extended_resnet34'
    model.compile(
        loss = 'categorical_crossentropy', 
        optimizer = optimizers.SGD(lr = 0.01, momentum = 0.9, decay = 0.0001),
        metrics=['accuracy'])

    return model
