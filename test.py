from i_want_video import GiveMeVideo
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import cv2 as cv
import numpy as np


vid = GiveMeVideo()
#vid.get_only_face()
vid.create_face_dataset('DataSet\\test\\rusin', 'rusin')

'''frame, k = vid.get_one_frame()
#cv.imwrite('DataSet\\test\\frolov.91.jpg', frame)
frame = cv.resize(frame, (224, 224))
cv.destroyAllWindows()
cv.imshow('Video', frame)
#img = image.load_img('DataSet\\test\\frolov.91.jpg', target_size=(224, 224))
x = image.img_to_array(frame)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
preds = model.predict(x)
print(decode_predictions(preds, top=3)[0])


train_dir = 'DataSet\\train'
test_dir = 'DataSet\\test'
epochs = 30
batch_size = 20
datagen = image.ImageDataGenerator(rescale=1. / 255)

train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical')

vgg19 = VGG19(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3))
vgg19.trainable = False
model = Sequential()
model.add(vgg19)
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(4))
model.add(Activation('sigmoid'))
# model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=1e-5),
              metrics=['accuracy'])
'''
