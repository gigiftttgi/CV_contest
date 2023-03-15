
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Flatten, Input, Dropout
from keras.layers import Conv2D, MaxPool2D
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np

IM_SIZE = 128

datagen = ImageDataGenerator(rescale=1./255)


train_generator = datagen.flow_from_directory(
    'dataset3/train',
    shuffle=True, 
    target_size=(IM_SIZE,IM_SIZE), 
    batch_size=50, 
    color_mode = 'rgb', 
    class_mode='categorical') 

validation_generator = datagen.flow_from_directory(
    'dataset3/validation',
    shuffle=False,
    target_size=(IM_SIZE,IM_SIZE),
    batch_size=50,
    color_mode='rgb',
    class_mode='categorical')

model = load_model('contest_model_3.h5')
# Train Model
# check point
checkpoint = ModelCheckpoint('contest_model_3.h5', verbose=1, monitor='val_accuracy',save_best_only=True, mode='max')

# ต้อว fit ผ่าน generator
h = model.fit_generator(
    train_generator,
    epochs=20,
    steps_per_epoch=len(train_generator),
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[checkpoint], verbose=1)
