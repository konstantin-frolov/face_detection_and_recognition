from keras.models import load_model
from keras.preprocessing import image

test_dir = 'DataSet\\test\\'
batch_size = 20
model = load_model('face_recognition.h5')
num_test_samples = 100

datagen = image.ImageDataGenerator(rescale=1. / 255)
test_gen = datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical')

scores = model.evaluate_generator(test_gen, num_test_samples // batch_size)

print('Точность на тестовых данных составляет: %.2f%%' % (scores[1]*100))
