import numpy as np
import pandas as pd

import os
batch_size = 64
epochs = 50
from PIL import Image
from keras.applications import ResNet50
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.layers import Lambda, Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, GaussianNoise
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils



x_train  = pd.read_csv("/media/nadeemqwerty/DATA/compiDL/train.csv")
print(x_train.shape)

y_train = x_train["label"]

y_train = to_categorical(y_train, num_classes=10)
print(y_train[0])
x_train = x_train.drop(labels = ["label"], axis=1)

print(x_train.shape)
x_train = x_train.values.reshape(-1,28,28,1)

print(x_train.shape)
x_train = x_train / 255.0

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.2, random_state = 29)

prev_model = ResNet50(weights='imagenet',include_top=False, input_shape=(28,28,1))
'''model.layers.pop()
model.layers.pop()
model.layers[-1].outbound_nodes = []
inp = model.input
opt = model.layers[-1].output
opt2 = Dense(10, activation = 'relu', name = 'fc_2')(opt)
##opt3 = Dense(512, activation = 'relu', name = 'fc_3')(opt2)
##opt4 = Dense(2, activation = 'softmax', name = 'fc_4')(opt3)
model = Model(input = inp, output = opt2)
'''
prev_model.layers.pop()
model = Sequential()
model.add(prev_model)
model.add(Dense(10, activation='softmax'))
print(model.summary())
'''model = Sequential()

model.add(Conv2D(filters = 32, kernel_size=(5,5), padding='Same', activation = 'relu', input_shape = (28,28,1)))
model.add(GaussianNoise(0.03))
##model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(filters = 32, kernel_size=(5,5), padding='Same',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(filters = 64, kernel_size=(3,3), padding='Same',activation='relu'))
model.add(BatchNormalization())
##model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(filters = 64, kernel_size=(3,3), padding='Same',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(filters = 128, kernel_size=(3,3), padding='Same',activation='relu'))
model.add(BatchNormalization())
##model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(filters = 128, kernel_size=(3,3), padding='Same',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

##model.add(Dense(128, activation='relu'))
##model.add(Dropout(0.5))
'''


adam = optimizers.Adam(lr = 3e-7)

datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=False,
    vertical_flip=False)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(x_train)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=x_train.shape[0] // batch_size,validation_data=(x_test, y_test), epochs=epochs, verbose = 1)

##print(model.summary)

test=pd.read_csv("/media/nadeemqwerty/DATA/compiDL//test.csv")

test = test.values.reshape(-1, 28, 28, 1)
test = test / 255.0

# predict results
print("start prediction")
results = model.predict(test)

# select the indix with the maximum probability
results = np.argmax(results, axis=1)

results = pd.Series(results, name="Label")
submission = pd.concat([pd.Series(range(1, 28001), name="ImageId"), results], axis=1)
print("write output")

submission.to_csv("submission.csv", index=False, header=True)
