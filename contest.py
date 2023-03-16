
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Flatten, Input, Dropout
from keras.layers import Conv2D, MaxPool2D
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np

# IM_SIZE = 64
IM_SIZE = 128

# model => 0.762499988079071
# model 2 => 0.7300000190734863
# model 3 => 0.7699999809265137
# model 4 => 0.7549999952316284 , 0.76250
# model 5 => 0.7524999976158142 , 0.70500
# model 6 => 0.7699999809265137 , 0.72000
# model 7 => 0.7325000166893005 , 0.68250
# model 8 => 0.7950000166893005 , 0.71000
# model 9 => 0.7450000047683716 , 0.73000
# model 10 => 0.7825000286102295 , 0.75250
# model 11 => 0.7799999713897705 , 0.75250
# model 12 => 0.7649999856948853 , 0.78500
# model 13 => 0.7799999713897705 , 0.75250

input = Input(shape = (IM_SIZE,IM_SIZE,3))

conv1 = Conv2D(32,3,activation='relu')(input)
pool1 = MaxPool2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(32,3,activation='relu')(pool1)
pool2 = MaxPool2D(pool_size=(2, 2))(conv2)

conv3 = Conv2D(32,3,activation='relu')(pool2)
pool3 = MaxPool2D(pool_size=(2, 2))(conv3)

conv4 = Conv2D(64,3,activation='relu')(pool3)
pool4 = MaxPool2D(pool_size=(2, 2))(conv4)

conv5 = Conv2D(128,3,activation='relu')(pool4)
pool5 = MaxPool2D(pool_size=(2, 2))(conv5)

flat = Flatten()(pool5)
hidden = Dense(64, activation='relu')(flat)
drop = Dropout(0.3)(hidden)
hidden = Dense(32, activation='relu')(hidden)
drop = Dropout(0.3)(hidden)
output = Dense(4, activation='softmax')(drop)
model = Model(inputs=input, outputs=output)


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


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

test_generator = datagen.flow_from_directory(
    'dataset3/test',
    shuffle=False,
    target_size=(IM_SIZE,IM_SIZE),
    batch_size=50,
    color_mode='rgb',
    class_mode='categorical')

# Train Model
# check point
checkpoint = ModelCheckpoint('contest_model_13.h5', verbose=1, monitor='val_accuracy',save_best_only=True, mode='max')

# ต้อว fit ผ่าน generator
h = model.fit_generator(
    train_generator,
    epochs=30,
    steps_per_epoch=len(train_generator),
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[checkpoint], verbose=1)

plt.plot(h.history['accuracy'])
plt.plot(h.history['val_accuracy'])
plt.legend(['train', 'val'])

# test model
model = load_model('contest_model_13.h5')
score = model.evaluate_generator(
    test_generator,
    steps=len(test_generator))
print('score (cross_entropy, accuracy):\n',score)

plt.show()
