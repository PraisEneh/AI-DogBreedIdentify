import random
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import sparse_categorical_crossentropy, categorical_crossentropy
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import cv2
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras import initializers
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

DATADIR = "A:/python/PyCharmProjects/pythonProject/big_data/training_images"
CATEGORIES = ['beagle', 'bloodhound', 'boston_bull', 'boxer', 'chihuahua', 'german_shepherd', 'golden_retriever',
              'maltese', 'rottweiler', 'saint_bernard']
#NAME = f'Three-Class-Test-{int(time.time())}'

IMG_SIZE = 64
training_data = []
now = datetime.now()


def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                new_array = np.array(new_array)
                training_data.append([new_array, class_num])
            except Exception as e:
                pass


create_training_data()

random.shuffle(training_data)

X = []
y = []
for features, label in training_data:
    X.append(features)
    y.append(label)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
y_train = np.array(y_train)

#find the link and explain this action
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
#print('Shape: ', X.shape[0])
print('Train Shape: ', X_train.shape)
# print(X.size)
# print(X.ndim)

# X.reshape((X.shape[0], -1))
X_train = np.array(X_train).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
X_train = X_train / (255 * 255 * 255)
# X = tf.stack(X)



POOL_SIZES = [2]
STRIDE_SIZES = []
KERNEL_SIZES = [3, 4, 6]
# how many dense layers we want to have
dense_layers = [1, 2, 3, 4]
layer_sizes = [64, 128, 256]
# how many conv layers do we want to have
conv_layers = [12]
# how many flattens do we want to have


learn_rates = [0.001, 0.003, 0.005]#, 0.05, 0.1, 0.3, 0.5]
dropout_rates = [0.2, 0.3]
filters = []

#use HeNormal initializer with relu function
active_functions = []
initializer = []

#
#GETTING RID OF VALIDATION AND ADDING SPLITTING THE DATASET BEFORE THE FIT METHOD
#
#

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            for learn_rate in learn_rates:
                for pool_size in POOL_SIZES:
                    for kernel_size in KERNEL_SIZES:
                        for dropout in dropout_rates:
                            current_time = now.strftime('%m-%d-%Y--%H-%M-%S')
                            NAME = f'{conv_layer}-conv-{layer_size}-nodes-{dense_layer}-dense-{learn_rate}-learn rate-{pool_size}-pool-No-stride-{kernel_size}-kernel-{dropout}-dropout-Softmax-function-{current_time}'
                            tb = TensorBoard(log_dir=f'logs/{NAME}')
                            optimize = SGD(learning_rate=learn_rate)
                            #init = each_initializer
                            print(NAME)

                            model = Sequential()
                                #maybe start by changing the input layer size
                            model.add(layers.Convolution2D(64, (kernel_size, kernel_size),
                                                           input_shape=X_train.shape[1:], activation='relu'))
                                #no stride
                            model.add(layers.MaxPooling2D(pool_size=(pool_size, pool_size),
                                                          ))

                            for l in range(conv_layer - 1):
                                model.add(layers.Convolution2D(layer_size, (kernel_size, kernel_size),
                                                               activation='relu', padding='same'))
                                    #no stride
                                model.add(layers.MaxPooling2D(pool_size=(pool_size, pool_size),
                                                              padding='same'))

                                #removed global pool2D avg
                            #for l in range(flatten_layer):
                            model.add(layers.Flatten())
                            model.add(layers.Flatten())
                            model.add(layers.Flatten())

                            for l in range(dense_layer):
                                model.add(layers.Dense(layer_size, activation='relu'))
                            model.add(layers.Dropout(dropout))
                                #no kernel init
                                #no active func
                            model.add(layers.Dense(10, activation='softmax'))

                            # Train
                            model.compile(loss=categorical_crossentropy, optimizer=optimize,
                                          metrics=['accuracy'])
                            model.fit(X_train, y_train, batch_size=32, epochs=50,
                                      callbacks=[tb], validation_data=(X_test, y_test))


