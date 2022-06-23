<<<<<<< HEAD
import random
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
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
NAME = f'Three-Class-Test-{int(time.time())}'

IMG_SIZE = 64
training_data = []
now = datetime.now()

print(len(tf.config.list_physical_devices('GPU')))

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



POOL_SIZES = [2, 4]
STRIDE_SIZES = [2]
KERNEL_SIZES = [16, 32, 64]
# how many dense layers we want to have
dense_layers = [2, 3, 4]
layer_sizes = [64, 128, 256]
# how many conv layers do we want to have
conv_layers = [5, 6, 8]

learn_rates = [0.001, 0.003, 0.01, 0.03] #, 0.05, 0.1, 0.3, 0.5]
dropout_rates = [0.2, 0.3]
filters = [str(8)]

#use HeNormal initializer with relu function
active_functions = ['softmax', 'sigmoid']
initializer = [initializers.GlorotNormal(), initializers.GlorotUniform()]

#
#GETTING RID OF VALIDATION AND ADDING SPLITTING THE DATASET BEFORE THE FIT METHOD
#
#

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
                                        for each_initializer in initializer:
                                            try:
                                                current_time = now.strftime('%H_%M_%S')
                                                NAME = f'{conv_layer}-conv-{layer_size}-nodes-{dense_layer}-dense-{learn_rate}-learn rate-{pool_size}-pool-{stride_size}-stride-{kernel_size}-kernel-{dropout}-dropout-{each_function}-function-{current_time}'
                                                tb = TensorBoard(log_dir=f'logs/{NAME}')
                                                optimize = Adam(learning_rate=learn_rate)
                                                init = each_initializer
                                                print(NAME)

                                                model = Sequential()
                                                model.add(layers.Convolution2D(64, (kernel_size, kernel_size),
                                                                               input_shape=X_train.shape[1:], activation='relu'))
                                                model.add(layers.MaxPooling2D(pool_size=(pool_size, pool_size),
                                                                              strides=(stride_size, stride_size)))

                                                for l in range(conv_layer - 1):
                                                    model.add(layers.Convolution2D(layer_size, (kernel_size, kernel_size),
                                                                                   activation='relu', padding='same'))

                                                    model.add(layers.MaxPooling2D(pool_size=(pool_size, pool_size),
                                                                                  strides=(stride_size, stride_size)))

                                                model.add(layers.GlobalAvgPool2D())
                                                for l in range(dense_layer):
                                                    model.add(layers.Dense(layer_size, activation='relu'))
                                                model.add(layers.Dropout(dropout))
                                                model.add(layers.Dense(10, activation=each_function, kernel_initializer=init))

                                                # Train
                                                model.compile(loss=categorical_crossentropy, optimizer=optimize,
                                                              metrics=['accuracy'])
                                                model.fit(X_train, y_train, batch_size=32, epochs=80,
                                                          callbacks=[tb], validation_data=(X_test, y_test))
                                            except Exception as e:
                                                print('exception')
                                                print(e.with_traceback())
                                                print(e.__traceback__)
                                                pass



"""
Test Data Results

3-conv-64-nodes-2-dense-1649025778: 29% accuracy
3-conv-32-nodes-0-dense-0.001-learn rate-2-pool-1-stride-4-kernel-0.3-dropout-softmax-function-23_21_35: 57% accuracy
3-conv-32-nodes-0-dense-0.001-learn rate-2-pool-1-stride-4-kernel-0.3-dropout-sigmoid-function-23_21_35: 49% accuracy
3-conv-32-nodes-0-dense-0.001-learn rate-2-pool-1-stride-4-kernel-0.3-dropout-21_42_26

'most successful combination so far: 3-conv-32-nodes-0-dense-0.001-learn rate-2-pool-1-stride-4-kernel-0.3-dropout'

combinations of interest: 
3-conv-32-nodes-0-dense-0.001-learn rate-4-pool-1-stride-4-kernel-0.3-dropout-sigmoid-function-23_21_35

"""
"""
Checkpoint
3-conv-32-nodes-0-dense-0.001-learn rate-2-pool-7-stride-8-kernel-0.1-dropout-20_53_01

3-conv-32-nodes-0-dense-0.001-learn rate-3-pool-5-stride-2-kernel-0.3-dropout-21_42_26

3-conv-32-nodes-0-dense-0.001-learn rate-3-pool-5-stride-3-kernel-0.2-dropout-21_42_26

3-conv-32-nodes-0-dense-0.001-learn rate-6-pool-3-stride-1-kernel-0.2-dropout-softmax-function-23_21_35


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
=======
import random
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy, binary_crossentropy, categorical_crossentropy, SparseCategoricalCrossentropy
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import cv2
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

DATADIR = "A:/python/PyCharmProjects/pythonProject/big_data/training_images"
CATEGORIES = ['bloodhound', 'golden_retriever', 'maltese']
NAME = f'Three-Class-Test-{int(time.time())}'

IMG_SIZE = 64
training_data = []
now = datetime.now()

print(len(tf.config.list_physical_devices('GPU')))

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

y_train = to_categorical(y_train, 3)
y_test = to_categorical(y_test, 3)
#print('Shape: ', X.shape[0])
print('Train Shape: ', X_train.shape)
# print(X.size)
# print(X.ndim)

# X.reshape((X.shape[0], -1))
X_train = np.array(X_train).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
X_train = X_train / (255 * 255 * 255)
#X = tf.stack(X)



POOL_SIZES = [i + 1 for i in range(10)]
STRIDE_SIZES = [i + 1 for i in range(10)]
KERNEL_SIZES = [i + 1 for i in range(10)]
# how many dense layers we want to have
dense_layers = [0, 1, 2]
layer_sizes = [32, 64, 128]
# how many conv layers do we want to have
conv_layers = [3, 4, 5, 6]

learn_rates = [0.001, 0.003, 0.005]  # , 0.01, 0.03, 0.05, 0.1, 0.3, 0.5]
dropout_rates = [0.1, 0.2, 0.3]
filters = [str(8)]

active_functions = ['softmax', 'tanh', 'sigmoid']

#
#GETTING RID OF VALIDATION AND ADDING SPLITTING THE DATASET BEFORE THE FIT METHOD
#
#

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
                                            optimize = Adam(learning_rate=learn_rate)
                                            print(NAME)

                                            model = Sequential()
                                            model.add(layers.Convolution2D(64, (kernel_size, kernel_size),
                                                                           input_shape=X_train.shape[1:], activation='relu'))
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
                                            model.add(layers.Dense(3, activation=each_function))

                                            # Train
                                            model.compile(loss=categorical_crossentropy, optimizer=optimize,
                                                          metrics=['accuracy'])
                                            model.fit(X_train, y_train, batch_size=32, epochs=30,
                                                      callbacks=[tb], validation_data=(X_test, y_test))
                                        except Exception as e:
                                            print('exception')
                                            print(e.with_traceback())
                                            print(e.__traceback__)
                                            pass



"""
Test Data Results

3-conv-64-nodes-2-dense-1649025778: 29% accuracy

"""
"""
Checkpoint
3-conv-32-nodes-0-dense-0.001-learn rate-2-pool-7-stride-8-kernel-0.1-dropout-20_53_01



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
>>>>>>> master
