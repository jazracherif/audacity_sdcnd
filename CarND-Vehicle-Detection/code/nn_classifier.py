
'''Train a simple deep CNN on the CIFAR10 small images dataset.
It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import matplotlib.image as mpimg
import glob
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from keras.layers.convolutional import Convolution2D
from keras.callbacks import Callback
import cv2

class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data
    
    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print('\n Training loss: {}, acc: {}\n'.format(loss, acc))

batch_size = 64
num_classes = 2
epochs = 10

data_augmentation = False

def load_data():
    """
       Load the image and split into train/dev/test data sets
    """
    # Divide up into cars and notcars
    if os.path.exists("./nn_data.pkl"):
        data = pickle.load(open("./nn_data.pkl", "rb"))
        X_train, y_train,  X_valid, y_valid, X_test, y_test = data["X_train"], data["y_train"], data["X_valid"], data["y_valid"], data["X_test"], data["y_test"]
        return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)
    
    else:
        CARS_FILES = ['../data/vehicles/GTI_Far/*.png',
                      '../data/vehicles/GTI_Left/*.png',
                      '../data/vehicles/GTI_MiddleClose/*.png',
                      '../data/vehicles/GTI_Right/*.png',
                      '../data/vehicles/KITTI_extracted/*.png']

        NOT_CARS_FILES = [ '../data/non-vehicles/GTI/*.png',
                          '../data/non-vehicles/Extras/*.png']

        cars = []
        notcars = []

        for f in CARS_FILES:
            cars += glob.glob(f)
        for f in NOT_CARS_FILES:
            notcars += glob.glob(f)

    #    sample_size = 500
    #    cars = cars[0:sample_size]
    #    notcars = notcars[0:sample_size]

        car_images = []
        not_car_images = []
        for f in cars:
            car_images.append(mpimg.imread(f))

        for f in notcars:
            not_car_images.append(mpimg.imread(f))

        print ("#cars, #notcars", len(car_images), len(not_car_images))
        X = np.vstack((car_images, not_car_images))
        y = np.hstack((np.ones(len(car_images)), np.zeros(len(not_car_images))))

        _X_train, X_test, _y_train, y_test = train_test_split(
                                                        X, y, test_size=0.1)

        X_train, X_valid, y_train, y_valid = train_test_split(
                                                    _X_train, _y_train, test_size=0.2)

        data = {"X_train":X_train, "X_valid":X_valid, "X_test":X_test, "y_train":y_train, "y_valid":y_valid, "y_test":y_test}

        pickle.dump(data, open("./nn_data.pkl", "wb+"))

        return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

# The data, shuffled and split between train and test sets:
#(x_train, y_train), (x_test, y_test) = cifar10.load_data()
(x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_data()


print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_valid.shape[0], 'valid samples')
print(x_test.shape[0], 'test samples')

##subimg_rgb = cv2.cvtColor(x_train[0], cv2.COLOR_HLS2RGB)
#mpimg.imsave('./out/testimage.jpg', x_train[100])

# Convert class vectors to binary class matrices.
y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
y_valid = keras.utils.np_utils.to_categorical(y_valid, num_classes)
y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)

def nn_architecture():
    """
       Build a Convnet.
       
       This is a simplified version of the cifar_10 model from
       https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py
    """
    
    model = Sequential()

    # Layer 1 - CONV_64
    model.add(Convolution2D(64, 5, 5, input_shape=(64, 64, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Layer 2 - CONV_64
    model.add(Convolution2D(64, 5, 5, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.8))

    model.add(Flatten())
    
    # LAYER 3 - FC_32
    model.add(Dense(32))
    model.add(Activation('relu'))

    # LAYER 4 - FC_2
    model.add(Dense(num_classes))
    
    # Final Layer - Softmax
    model.add(Activation('softmax'))

    return model

model = nn_architecture()

# initiate Adam optimizer
opt = keras.optimizers.Adam(lr=0.001)

# Let's train the model using Adam
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_valid = x_valid.astype('float32')
x_test = x_test.astype('float32')

# Rescale if not png
PNG = False
if PNG:
    print ("Images are PNG, don't rescale")
    x_train /= 255
    x_valid /= 255
    x_test /= 255

if not data_augmentation:
    print('Not using data augmentation.')
    
    model.fit(x_train, y_train,
              batch_size=batch_size,
              nb_epoch=epochs,
              validation_data=(x_valid, y_valid),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        workers=4)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
