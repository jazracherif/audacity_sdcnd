import pandas as pd
import numpy as np
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import csv
import cv2
import os

import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt
import random

import keras
import tensorflow as tf

from keras.layers.convolutional import Convolution2D

from keras.layers import Input, Flatten, Dense, Activation, Lambda, Cropping2D, MaxPooling2D, Dropout
from keras.models import Model
from keras.models import Sequential
from keras import initializations

DIR = "./training_udacity/"

def generator(samples, batch_size=32):
    """
        A function fed to the Keras fit function to pull the dataset
        from the file system using python generators, therefore saving
        time and space.
    """
    
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            for batch_sample in batch_samples:
                
                name = "./training_udacity/IMG/"+batch_sample[0].strip().split('/')[-1]

                center_image = cv2.imread(name)
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)

                center_angle = float(batch_sample[3])
                throttle = float(batch_sample[4])
                brake = float(batch_sample[5])
                speed = float(batch_sample[6])
                
                images.append(center_image)
                angles.append(center_angle)
            
                augment = True
                if augment:
                    # 1. Add Flipped Picture
                    image_flipped = np.fliplr(center_image)
                    measurement_flipped = -center_angle
                    
                    images.append(image_flipped)
                    angles.append(measurement_flipped)
        
                    # 2. Handle left and right Images
                    # create adjusted steering measurements for the side camera images
                    correction = 0.4
                    steering_left = center_angle + correction
                    steering_right = center_angle - correction
                    
                    left_name = "./training_udacity/IMG/"+batch_sample[1].strip().split('/')[-1]
                    right_name = "./training_udacity/IMG/"+batch_sample[2].strip().split('/')[-1]

                    img_left = cv2.imread(left_name)
                    img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)

                    img_right = cv2.imread(right_name)
                    img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)

                    images.append(img_left)
                    images.append(img_right)
                    
                    angles.append(steering_left)
                    angles.append(steering_right)

# Sanity check the code above by plotting each picture
#                fig = plt.figure()
#                plt.imshow(center_image)
#                plt.axis('off')
#                fig.savefig("center.jpg")
#
#                fig = plt.figure()
#                plt.imshow(image_flipped)
#                plt.axis('off')
#                fig.savefig("flipped.jpg")
#
#                fig = plt.figure()
#                plt.imshow(img_left)
#                plt.axis('off')
#                fig.savefig("left.jpg")
#
#                fig = plt.figure()
#                plt.imshow(img_right)
#                plt.axis('off')
#                fig.savefig("right.jpg")

            X_train = np.array(images)
            y_train = np.array(angles)
            
            yield shuffle(X_train, y_train)



flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_file', '', "Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file', '', "Bottleneck features validation file (.p)")
flags.DEFINE_integer('epochs', 50, "The number of epochs.")
flags.DEFINE_integer('batch_size', 256, "The batch size.")
flags.DEFINE_float('learning_rate', 0.01, "Set the Learning rate.")
flags.DEFINE_boolean('resume', False, "Whether to resume from last run. This will load the model in model.h5")
flags.DEFINE_boolean('test', False, "Test the model on a small dataset.")
flags.DEFINE_boolean('predict', False, "Predict using the learned model using a small dataset, typically for sanity check.")

batch_size = FLAGS.batch_size
epochs = FLAGS.epochs 
learning_rate = FLAGS.learning_rate
resume = FLAGS.resume
test = FLAGS.test
predict = FLAGS.predict


# 1. Load the data
samples = []
filelist = os.listdir('./training_udacity/')
for f in filelist:
    if "unbiased_driving_log" in f:
        with open('./training_udacity/'+f) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                samples.append(line)


if test:
    print ("Test on Small Dataset")
    samples = shuffle(samples)
    samples = samples[:5]

# 2. Split for training and validation
train_samples, validation_samples = train_test_split(samples, test_size=0.1)

print ("train data {} , valid data {}".format(len(train_samples), len(validation_samples)))

# 3. Create Generators for loading the data 1 batch at a time
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

def my_init(shape, name=None):
    """
        A custom initialization function for model parameters
    """
    return initializations.normal(shape, scale=0.01, name=name)

def create_simple_model():
    """
        A Simple Model to verify the good functioning of the code
    """

    input_shape = (160, 320, 3)
    
    m = Sequential()

    # 1. Add Normalization
    m.add(Lambda(lambda x: x/255.0 - 0.5,
                 input_shape=input_shape,
                 ))

    # 2. Flatten + 1 fully connected layer
    m.add(Flatten())
    m.add(Dense(10, activation='relu', init=my_init))
    
    # 3. Output Layer is a Dense layer with no activation function
    m.add(Dense(1))
                 
    return m

def create_mnist_model():
    """
        MNIST Conv model inspired by
        https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
    """

    input_shape = (160, 320, 3)
    
    m = Sequential()
    
    new_size = [64,64]
    def mylambda(x):
        import tensorflow as tf
        return tf.image.resize_images(x, size=(64,64))
    
    # 1. Resize Input to 64x64
    m.add(Lambda(mylambda,
                 input_shape=input_shape,
                 ))
                 
    # 2. Normalize
    m.add(Lambda(lambda x: x/255.0 - 0.5,
                 ))
           
    # 3. Add 2 Conv layers with max pooling, and dropouts
    m.add(Convolution2D(5, 3, 3, subsample=(1,1), activation='relu'))
    m.add(Convolution2D(10, 3, 3, subsample=(1,1), activation='relu'))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Dropout(0.25))
    
    # 4. Flatten and use Fully Connected module
    m.add(Flatten())
    m.add(Dense(50, activation='relu'))
    m.add(Dropout(0.5))
    m.add(Dense(1, activation=None))
    
    return m

def create_advanced_model():
    """
       More Advanced Model inspired by NVIDIA end to end model in https://arxiv.org/pdf/1604.07316.pdf
    """
    
    input_shape = (160, 320, 3)

    m = Sequential()

    # 1. Crop all pictures
    m.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=input_shape))

    # 2. Resize pictures to 64x64
    new_size = [64,64]
    def mylambda(x):
        import tensorflow as tf
        return tf.image.resize_images(x, size=(64,64))
    
    m.add(Lambda(mylambda,
                 ))

    # 3. Normalize
    m.add(Lambda(lambda x: x/255- 0.5,
                     input_shape=(64, 64, 3),
                     output_shape=(64, 64, 3)))


    # 4. Three Conv layers with strides 2
    m.add(Convolution2D(24, 5, 5, subsample=(2,2), init='he_normal'))
    m.add(Activation('relu'))

    m.add(Convolution2D(36, 5, 5, subsample=(2,2), init='he_normal' ))
    m.add(Activation('relu'))

    m.add(Convolution2D(48, 5, 5, subsample=(2,2), init='he_normal'))
    m.add(Activation('relu'))

    # 5. Two Conv layers with stride 1
    m.add(Convolution2D(64, 3, 3, init='he_normal'))
    m.add(Activation('relu'))

    m.add(Convolution2D(64, 3, 3, init='he_normal'))
    m.add(Activation('relu'))
    #m.add(Dropout(0.5))

    # 6. Flatten and add Fully Connected Layers
    m.add(Flatten())
    m.add(Dense(1164, init='he_normal', activation='relu'))
    m.add(Dense(100, init='he_normal', activation='relu'))
    m.add(Dense(50, init='he_normal', activation='relu'))
    m.add(Dense(10, activation='relu'))
    m.add(Dense(1, activation=None))

    return m

if predict:
    """
        Do predict on a small dataset
    """
    print ("PREDICT - Loading Model..")
    m = keras.models.load_model("./model.h5")

    samples = shuffle(samples)
    batch_sample = samples[:50]

    for s in batch_sample:

        name = "./training_udacity/IMG/"+s[0].strip().split('/')[-1]
        center_image = cv2.imread(name)
        center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)

        val  = m.predict(center_image[None, :, :, :], batch_size=1)
        
        predicted_angle = float(val)
        center_angle = float(s[3])

        print ("true angle {}, predicted angle {}".format(center_angle, predicted_angle))
else:
    """
        Do Training
    """
    print ("Training..")
    if resume:
        print ("Resuming..")
        m = keras.models.load_model("./model.h5")
    else:
        print ("Create New Model..")
        m = create_mnist_model()

    opt = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    m.compile(loss='mse',
              optimizer= opt,
              metrics=['mae'])
    history_object = m.fit_generator(train_generator,
                              samples_per_epoch=4*len(train_samples),
                              validation_data=validation_generator,
                              nb_val_samples=4*len(validation_samples),
                              nb_epoch=epochs,
                              verbose=1,
                              )
    # Save the model to disk
    m.save("./model.h5")

    # Draw histograms of losses for training and validation set
    print ("loss", history_object.history['loss'])

    fig = plt.figure()
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    #plt.show()
    fig.savefig("./results/loss.jpg")

print ("DONE!")
