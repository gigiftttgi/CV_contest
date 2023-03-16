from keras.models import Model, load_model
from keras.layers import Dense, Flatten, Input
from keras.layers import Conv2D, MaxPool2D
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

IM_SIZE = 128

model = load_model('modelAll/contest_model_10.h5')

c = os.listdir("contestData")

className = {0:'B',1:'D',2:'R',3:'S'}

result = open("result.txt", "w")
for n in c:
    im = cv2.imread("contestData/"+ n,cv2.IMREAD_COLOR)
    im = cv2.resize(im,(128,128))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im/255.
    im = np.expand_dims(im, axis=0)
    predict = model.predict(im)
    classImg = np.argmax(predict,axis = -1)[0]
    result.write(str(n) + "::" + className[classImg]  + "\n")
