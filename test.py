from keras.models import Model, load_model
from keras.layers import Dense, Flatten, Input
from keras.layers import Conv2D, MaxPool2D
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np

IM_SIZE = 128


datagen = ImageDataGenerator(rescale=1./255)

test_generator = datagen.flow_from_directory(
    'dataset3/test',
    shuffle=False,
    target_size=(IM_SIZE,IM_SIZE),
    batch_size=50,
    color_mode='rgb',
    class_mode='categorical')

model = load_model('contest_model_3.h5')

score = model.evaluate_generator(
    test_generator,
    steps=len(test_generator))
print('score (cross_entropy, accuracy):\n',score)

predict = model.predict_generator(
    test_generator,
    steps=len(test_generator),
    workers = 1,
    use_multiprocessing=False)

className = ['B','D','R','S']
result = open("result.txt", "w")
predict_class_idx = np.argmax(predict,axis = -1)

predict_class_name = [className[x] for x in predict_class_idx]

f = test_generator.filenames
for i in range(len(f)) :
    result.write(str(f[i]) + ":" + predict_class_name[i]  + "\n")

result.close()


