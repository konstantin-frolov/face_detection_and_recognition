from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.optimizers import Adam
from keras.applications.resnet50 import preprocess_input, decode_predictions
import cv2 as cv
import numpy as np

#frame, k = vid.get_one_frame()
#cv.imwrite('DataSet\\test\\frolov.91.jpg', frame)
#frame = cv.resize(frame, (224, 224))
#cv.destroyAllWindows()
#cv.imshow('Video', frame)
#img = image.load_img('DataSet\\test\\frolov.91.jpg', target_size=(224, 224))

train_dir = 'DataSet\\train'
test_dir = 'DataSet\\test'
val_dir = 'DataSet\\val'
epochs = 30
batch_size = 20
num_train_samples = 899
num_test_samples = 100
num_val_samples = 50

datagen = image.ImageDataGenerator(rescale=1. / 255)

train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical')

test_gen = datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical')

val_gen = datagen.flow_from_directory(
    val_dir,
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
model.add(Dense(3))
model.add(Activation('sigmoid'))
#model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=1e-5),
              metrics=['accuracy'])

model.fit_generator(train_gen, steps_per_epoch=num_train_samples // batch_size,
                    epochs=epochs, validation_data=val_gen, validation_steps=num_val_samples // batch_size)

model.save('face_recognition.h5')
