# EMNIST Letter Dataset 88.72%:acc & val_acc: 87.81%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout


train_images = pd.read_csv("emnist/emnist-balanced-train.csv", header=None)
test_images = pd.read_csv("emnist/emnist-balanced-test.csv", header=None)
map_images = pd.read_csv("emnist/emnist-balanced-mapping.txt", header=None)

train_x = train_images.iloc[:, 1:]
train_y = train_images.iloc[:, 0]
train_x = train_x.values

test_x = test_images.iloc[:, 1:]
test_y = test_images.iloc[:, 0]
test_x = test_x.values


ascii_map = []
for i in map_images.values:
    ascii_map.append(i[0].split()[1])

def rot_flip(image):
    image = image.reshape([28, 28])
    image = np.fliplr(image)
    image = np.rot90(image)
    return image

train_x = np.apply_along_axis(rot_flip, 1, train_x)
test_x = np.apply_along_axis(rot_flip, 1, test_x)
plt.imshow(train_x[2])
train_x.shape

train_x = train_x.astype('float32')
train_x = train_x/255.0
test_x = test_x.astype('float32')
test_x = test_x/255.0

train_x = train_x.reshape(-1, 28,28, 1)   #Equivalent to (112800,28,28,1)
test_x = test_x.reshape(-1, 28,28, 1)   #Equivalent to (18800,28,28,1)

# Define the model
model = Sequential()
model.add(Conv2D(32,(3,3),input_shape = (28,28,1),activation = 'relu'))
model.add(Conv2D(64,(3,3),activation = 'relu'))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(47,activation='softmax'))

# Compile the model / Компиляция модели
model.compile(optimizer = 'adam',loss= "sparse_categorical_crossentropy", metrics=['accuracy'])
model.summary()

# Train the model / Обучение модели
history = model.fit(
    train_x,
    train_y,
    validation_data = (test_x,test_y),
    # how many epochs will be trained
    epochs = 20,
)

# Save the model to a file / Сохранение модели
model.save('emnist_model.keras')