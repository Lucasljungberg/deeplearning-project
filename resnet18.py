from resnetBuild import ResnetBuilder
from keras import optimizers

def create_model (shape = (32, 32, 3), classifiers = 10, train = False):
    model = ResnetBuilder.build_resnet_18(shape, classifiers)
    model.name = 'extended_resnet18'
    model.compile(
        loss = 'categorical_crossentropy', 
        optimizer = optimizers.SGD(lr = 0.001, momentum = 0.9),
        metrics=['accuracy'])

    return model
