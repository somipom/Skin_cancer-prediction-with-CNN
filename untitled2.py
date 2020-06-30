import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

classifier = Sequential()
classifier.add(Convolution2D(32, 3, 3, activation = 'relu', input_shape = (120,120,3)))
classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Convolution2D(32,3,3,activation='relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Flatten())

classifier.add(Dense(output_dim=128,activation='relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss= 'binary_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,vertical_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory  (
        'Skin cancer/train',
        target_size=(120, 120),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'Skin cancer/test',
        target_size=(120, 120),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        train_set,
        samples_per_epoch=2635,
        nb_epoch=50,
        validation_data=test_set,
        nb_val_samples=600)

import numpy as np
from keras.preprocessing import image
train_set.class_indices
test_image = image.load_img('Skin cancer/single Prediction/Benign or m.jpg',target_size = (120,120))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis = 0)
result = classifier.predict(test_image)
if result [0][0] == 1:
    prediction = 'malignant'
else:
    prediction = 'benign'










