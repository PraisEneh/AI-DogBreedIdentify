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
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

DATADIR = "A:/python/PyCharmProjects/pythonProject/big_data/training_images"
CATEGORIES = ['beagle', 'bloodhound', 'boston_bull', 'boxer', 'chihuahua', 'german_shepherd', 'golden_retriever',
              'maltese', 'rottweiler', 'saint_bernard']
NAME = f'Three-Class-Test-{int(time.time())}'

IMG_SIZE = 75
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
preprocess_input(X_train)
preprocess_input(X_test)


split_ratio = [85, 125, 190, 230, 280, 315, 356, 380]
learning_rates = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]
layer_densities = [32, 64, 92, 123]

#
#GETTING RID OF VALIDATION AND ADDING SPLITTING THE DATASET BEFORE THE FIT METHOD
#
#
for each_split_ratio in split_ratio:
    for each_learning_rate in learning_rates:
        current_time = now.strftime('%m-%d-%Y--%H-%M-%S')
        NAME = f'Inception-SGD-{each_split_ratio}-split ratio-{each_learning_rate}-learning rate' \
               f'-No-node density layer-{current_time}'
        tb = TensorBoard(log_dir=f'logs/{NAME}')
        optimize = SGD(learning_rate=each_learning_rate)
        print(NAME)

        # creating the base model using pretrained weights and custom input tensor
        print(X_train.shape[1:])
        base_model = InceptionResNetV2(input_shape=X_train.shape[1:], weights='imagenet', include_top=False)
        z = base_model.output
        z = layers.GlobalAvgPool2D()(z)
        # removed dense layer to pretrained model
        # 'logistic' layer -- lets say we have 10 classes
        output = layers.Dense(10, activation='softmax')(z)

        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=output)

        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        for layer in base_model.layers:
            layer.trainable = False


        # Train
        model.compile(loss=categorical_crossentropy, optimizer=optimize, metrics=['accuracy'])
        model.fit(X_train, y_train, batch_size=32, epochs=50, callbacks=[tb], validation_data=(X_test, y_test))

        # visualize layers, so we know how many to freeze
        #for i, layer in enumerate(base_model.layers):
        #    print(i, layer.name)

        for layer in model.layers[:each_split_ratio]:
            layer.trainable = False
        for layer in model.layers[each_split_ratio:]:
            layer.trainable = True
        tb = TensorBoard(log_dir=f'logs/{NAME} 2nd train')
        model.compile(loss=categorical_crossentropy, optimizer=optimize, metrics=['accuracy'])
        model.fit(X_train, y_train, batch_size=32, epochs=100, callbacks=[tb], validation_data=(X_test, y_test))




#model.add(layers.Convolution2D(64, (3, 3), input_shape=X_train.shape[1:], activation='relu', padding='same'))
#model.add(layers.GlobalAvgPool2D())

#model.add(layers.Activation('softmax'))

