import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras import utils
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
%matplotlib inline

batch_size=256
image_size=(100, 100)

train_dataset = image_dataset_from_directory('Gunmen/All',
                                            subset='training',
                                            seed=42,
                                            validation_split=0.1,
                                            batch_size=batch_size,
                                            image_size=image_size)

validation_dataset = image_dataset_from_directory('Gunmen/All',
                                            subset='validation',
                                            seed=42,
                                            validation_split=0.1,
                                            batch_size=batch_size,
                                            image_size=image_size)

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
class_names = train_dataset.class_names

test_dataset = image_dataset_from_directory('Gunmen/All',
                                            batch_size=batch_size,
                                            image_size=image_size)

model = Sequential()
model.add(Conv2D(16, (5, 5), padding='same',
                 input_shape=(100, 100, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (5, 5), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (5, 5), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(131, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])

history = model.fit(train_dataset,
                    validation_data=validation_dataset,
                    epochs=6,
                    verbose=2)

scores = model.evaluate(test_dataset, verbose=1)

model.save("gunmen.h5")

    