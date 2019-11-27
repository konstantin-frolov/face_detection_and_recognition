from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.optimizers import Adam, SGD
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
epochs = 3
batch_size = 17
num_train_samples = 899
num_test_samples = 100
num_val_samples = 100
INPUT_SHAPE = (160, 160, 3)
datagen = image.ImageDataGenerator(rescale=1. / 255)

train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=(160, 160),
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    seed=42)

test_gen = datagen.flow_from_directory(
    test_dir,
    target_size=(160, 160),
    color_mode='rgb',
    batch_size=1,
    class_mode='categorical',
    shuffle=True,
    seed=42
)

val_gen = datagen.flow_from_directory(
    val_dir,
    target_size=(160, 160),
    color_mode='rgb',
    batch_size=1,
    class_mode='categorical',
    shuffle=True,
    seed=42
)

# import vgg19
vgg19 = VGG19(
    weights='imagenet',
    include_top=False,
    input_shape=INPUT_SHAPE)
vgg19.trainable = False
#vgg19.summary()

# import resnet50
resnet = ResNet50(include_top=False,
                  pooling='avg',
                  weights='imagenet',
                  input_shape=INPUT_SHAPE)
sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
resnet.trainable = True

# import facenet
facenet = load_model('keras-facenet\\model\\facenet_keras.h5')
facenet.load_weights('keras-facenet\\weights\\facenet_keras_weights.h5')
facenet.trainable = False
facenet.summary()
'''trainable = False
for layer in vgg19.layers:
    if layer.name == 'block3_conv1':
        trainable = True
    layer.trainable = trainable
'''

model = Sequential()
model.add(facenet)
# model.add(resnet)
# model.add(vgg19)
# model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(3))
model.add(Activation('sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit_generator(train_gen, steps_per_epoch=num_train_samples // batch_size,
                    epochs=epochs, validation_data=val_gen, validation_steps=num_val_samples)

model.save('face_recognition_ep=10_facenet_with_conv.h5', include_optimizer=False)

scores = model.evaluate_generator(test_gen, num_test_samples)

print('Точность на тестовых данных составляет: %.2f%%' % (scores[1]*100))
