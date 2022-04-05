import random
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy, binary_crossentropy
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import cv2
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import backend as k

DATADIR = "A:/python/PyCharmProjects/pythonProject/big_data/training_images"
CATEGORIES = ['bloodhound', 'golden_retriever', 'maltese']
NAME = f'Three-Class-Test-{int(time.time())}'

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
X = np.array(X)
X.reshape((X.shape[0], -1))
#X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X = X / (255 * 255 * 255)
X = tf.stack(X)
y = np.array(y)
y = tf.stack(y)

print('shape: ', X.shape, y.shape)

POOL_SIZES = [i+1 for i in range(10)]
STRIDE_SIZES = [i+1 for i in range(10)]
KERNEL_SIZES = [i+1 for i in range(10)]
# how many dense layers we want to have
dense_layers = [0, 1, 2]
layer_sizes = [32, 64, 128]
# how many conv layers do we want to have
conv_layers = [3, 4, 5, 6]

learn_rates = [0.001, 0.003, 0.005]#, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5]
dropout_rates = [0.1, 0.2, 0.3]
filters = [str(8)]

active_functions = ['softmax', 'tanh', 'sigmoid']

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            for learn_rate in learn_rates:
                for pool_size in POOL_SIZES:
                    for stride_size in STRIDE_SIZES:
                        for kernel_size in KERNEL_SIZES:
                            for each_filter in filters:
                                for dropout in dropout_rates:
                                    for each_function in active_functions:
                                        try:
                                            current_time = now.strftime('%H_%M_%S')
                                            NAME = f'{conv_layer}-conv-{layer_size}-nodes-{dense_layer}-dense-{learn_rate}-learn rate-{pool_size}-pool-{stride_size}-stride-{kernel_size}-kernel-{dropout}-dropout-{current_time}'
                                            tb = TensorBoard(log_dir=f'logs/{NAME}')
                                            print(NAME)

                                            model = Sequential()
                                            model.add(layers.Convolution2D(64, (kernel_size, kernel_size),
                                                                           input_shape=X.shape[1:], activation='relu'))
                                            model.add(layers.MaxPooling2D(pool_size=(pool_size, pool_size),
                                                                          strides=(stride_size, stride_size)))

                                            for l in range(conv_layer - 1):
                                                model.add(layers.Convolution2D(layer_size, (kernel_size, kernel_size),
                                                                               activation='relu', padding='same'))

                                                model.add(layers.MaxPooling2D(pool_size=(pool_size, pool_size),
                                                                              strides=(stride_size, stride_size)))

                                            model.add(layers.Flatten())
                                            for l in range(dense_layer):
                                                model.add(layers.Dense(layer_size, activation='relu'))

                                            model.add(layers.Dropout(dropout))
                                            model.add(layers.Dense(1, activation=each_function))
                                            optimize = Adam(learning_rate=learn_rate)

                                            # Train
                                            model.compile(loss=binary_crossentropy, optimizer=optimize,
                                                          metrics=['accuracy'])
                                            model.fit(X, y, batch_size=32, epochs=30, validation_split=0.3, callbacks=[tb])
                                        except Exception as e:
                                            print('exception')
                                            print(e.with_traceback())
                                            pass

"""
Test Data Results

3-conv-64-nodes-2-dense-1649025778: 29% accuracy

"""

"""
train_ds = tf.keras.utils.image_dataset_from_directory(
    directory='big_data/training_images/',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(64,64)
)


valid_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory='big_data/validation_images/',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(64, 64)

)
"""
