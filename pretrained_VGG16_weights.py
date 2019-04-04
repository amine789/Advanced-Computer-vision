# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 02:24:57 2019

@author: amine bahlouli
"""

from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input
# from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

from glob import glob

IMAGE_SIZE = [100,100]
batch_sz=50
epochs = 20

train_path = 'C:/Users/amine bahlouli/Documents/fruits/fruits-360-small/Training'
valid_path = 'C:/Users/amine bahlouli/Documents/fruits/fruits-360-small/Validation'

image_files = glob(train_path+'/*/*.jp*g')
valid_image_files = glob(valid_path+'/*/*.jp*g')

folders = glob(train_path + '/*')

vgg =VGG16(input_shape= IMAGE_SIZE +[3], weights='imagenet',include_top=False)

for layer in vgg.layers:
    vgg.trainable=False
x = Flatten()(vgg.output)
prediction = Dense(len(folders), activation = 'softmax')(x)
model = Model(inputs=vgg.input,outputs=prediction)
model.summary()
model.compile(
  loss='categorical_crossentropy',
  optimizer='rmsprop',
  metrics=['accuracy']
)

gen = ImageDataGenerator(
  rotation_range=20,
  width_shift_range=0.1,
  height_shift_range=0.1,
  shear_range=0.1,
  zoom_range=0.2,
  horizontal_flip=True,
  vertical_flip=True,
  preprocessing_function=preprocess_input
)



test_gen = gen.flow_from_directory(valid_path, target_size=IMAGE_SIZE)
print(test_gen.class_indices)
labels = [None] * len(test_gen.class_indices)

for v,k in test_gen.class_indices.items():
    labels[k]=v

train_generator = gen.flow_from_directory(train_path,
                                          target_size=IMAGE_SIZE,
                                          shuffle=True,
                                          batch_size=batch_sz)
valid_generator = gen.flow_from_directory(
  valid_path,
  target_size=IMAGE_SIZE,
  shuffle=True,
  batch_size=batch_sz,
)

r = model.fit_generator(train_generator, validation_data=valid_generator,epochs=epochs,
                        steps_per_epoch = len(image_files)//batch_sz,validation_steps = len(valid_image_files)//batch_sz)
def get_confusion_matrix(data_path,N):
    predictions = []
    targets = []
    i=0
    for x,y in gen.flow_from_directory(data_path, target_size=IMAGE_SIZE,shuffle=False,batch_size= batch_sz**2):
        i+=1
        if i%50:
            print(i)
        p = model.predict(x)
        p = np.argmax(p,1)
        y = np.argmax(y,1)
        predictions = np.concatenate((predictions,p))
        targets = np.concatenate((targets,y))
        if len(targets)>N:
            break
        cm = confusion_matrix(predictions, targets)
        return cm
cm = get_confusion_matrix(train_path, len(image_files))
print(cm)

