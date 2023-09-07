# MNIST Letter Dataset 99.47%:acc & val_acc: 99.1%

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from tensorflow import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten, Convolution2D, MaxPooling2D

# Load the MNIST dataset / Загрузка набора данных MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data / Нормализация данных
x_train = x_train / 255
x_test = x_test / 255

x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Define the model
model = keras.Sequential([
    Convolution2D(32, (3,3), padding="same", activation="relu", input_shape=(28, 28, 1)),
    MaxPooling2D((2,2), strides=2),
    Convolution2D(64, (3,3), padding="same", activation="relu"),
    MaxPooling2D((2,2), strides=2),         
    Flatten(),
    Dense(128, activation="relu"),
    Dense(10, activation="softmax")
])

# print(model.summary())

# Compile the model / Компиляция модели
model.compile(optimizer='adam',
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])

# Train the model / Обучение модели
his = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_split=0.2)
model.evaluate(x_test, y_test)

# Save the model to a file / Сохранение модели
model.save('mnist_model.keras')