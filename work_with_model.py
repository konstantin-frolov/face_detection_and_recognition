import os
import shutil
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import image
from keras.optimizers import SGD


class WorkWithModel:
    # Method for creating train, val and test directories in portions 80% : 10% : 10% respectively
    # inputs: dir_name - path to directory where you saved all images using GiveMeVideo.create_face_dataset
    #         class_names - class names into which you split the data
    # outputs: nothing
    @staticmethod
    def create_test_val_train_data(dir_name, class_names):
        train_dir = 'train'
        val_dir = 'val'
        test_dir = 'test'
        test_portion = 0.1
        val_portion = 0.1
        WorkWithModel.__create_dirs(os.path.join(dir_name, train_dir))
        WorkWithModel.__create_dirs(os.path.join(dir_name, test_dir))
        WorkWithModel.__create_dirs(os.path.join(dir_name, val_dir))
        for i in range(len(class_names)):
            WorkWithModel.__create_dirs(os.path.join(dir_name, train_dir, class_names[i]))
            WorkWithModel.__create_dirs(os.path.join(dir_name, test_dir, class_names[i]))
            WorkWithModel.__create_dirs(os.path.join(dir_name, val_dir, class_names[i]))
            nb_imgs = len(os.listdir(os.path.join(dir_name, 'all_data', class_names[i])))
            index_list = list(range(1, nb_imgs + 1))
            np.random.shuffle(index_list)
            WorkWithModel.__copy_imgs(0, test_portion * nb_imgs, index_list, os.path.join(dir_name, 'all_data',
                                      class_names[i]), os.path.join(dir_name, test_dir, class_names[i]), class_names[i])
            WorkWithModel.__copy_imgs(test_portion * nb_imgs + 1, test_portion * nb_imgs + 1 + val_portion * nb_imgs,
                                      index_list, os.path.join(dir_name, 'all_data', class_names[i]),
                                      os.path.join(dir_name, val_dir, class_names[i]), class_names[i])
            WorkWithModel.__copy_imgs(test_portion * nb_imgs + 1 + val_portion * nb_imgs, nb_imgs,
                                      index_list, os.path.join(dir_name, 'all_data', class_names[i]),
                                      os.path.join(dir_name, train_dir, class_names[i]), class_names[i])

    # Private method for creating one directory
    @staticmethod
    def __create_dirs(dir_name):
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
        os.mkdir(dir_name)

    # Private method for coping part of images from source folder to destination folder
    @staticmethod
    def __copy_imgs(start_index, end_index, index_list, source_dir, dest_dir, class_name):
        for i in range(int(start_index), int(end_index)):
            shutil.copy2(os.path.join(source_dir, class_name + '.' + str(index_list[i]) + '.jpg'), dest_dir)

    # Method for renaming all files in directory to 'file_name_count_value.jpg'
    @staticmethod
    def rename_all_files(dir_name, new_files_name):
        count = 1
        for filename in os.listdir(dir_name):
            new_names = dir_name + new_files_name + str(count) + '.jpg'
            src = dir_name + filename
            os.rename(src, new_names)
            count += 1

    # Method for training neuronet
    # inputs: path2train_dir - path to dir with train data
    #         path2test_dir - path to dir with test data
    #         path2val_dir - path to dir with validation data
    #         epochs - numbers epochs for training
    #         batch_size - size of batch for training
    #         learn_rate - learning rate for optimizer
    #         save_model_name - name of model for saving
    # output: nothing
    @staticmethod
    def train_network(path2train_dir, path2test_dir, path2val_dir, epochs, batch_size, learn_rate, save_model_name):
        names = os.listdir(path2test_dir)
        num_train_samples = len(os.listdir(os.path.join(path2train_dir, names[0])))
        num_test_samples = len(os.listdir(os.path.join(path2test_dir, names[0])))
        num_val_samples = len(os.listdir(os.path.join(path2val_dir, names[0])))
        INPUT_SHAPE = (160, 160, 3)
        datagen = image.ImageDataGenerator(rescale=1. / 255)

        train_gen = datagen.flow_from_directory(
            path2train_dir,
            target_size=(INPUT_SHAPE[0], INPUT_SHAPE[1]),
            color_mode='rgb',
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True,
            seed=42)

        test_gen = datagen.flow_from_directory(
            path2test_dir,
            target_size=(INPUT_SHAPE[0], INPUT_SHAPE[1]),
            color_mode='rgb',
            batch_size=1,
            class_mode='categorical',
            shuffle=True,
            seed=42
        )

        val_gen = datagen.flow_from_directory(
            path2val_dir,
            target_size=(INPUT_SHAPE[0], INPUT_SHAPE[1]),
            color_mode='rgb',
            batch_size=1,
            class_mode='categorical',
            shuffle=True,
            seed=42
        )

        sgd = SGD(lr=learn_rate, decay=1e-6, momentum=0.9, nesterov=True)

        # import facenet
        facenet = load_model('keras-facenet\\model\\facenet_keras.h5')
        facenet.load_weights('keras-facenet\\weights\\facenet_keras_weights.h5')
        facenet.trainable = True
        trainable = False
        for layer in facenet.layers:
            if layer.name == 'Block17_5_Branch_1_Conv2d_0a_1x1':
                trainable = True
            layer.trainable = trainable

        model_net = Sequential()
        model_net.add(facenet)
        model_net.add(Dense(256))
        model_net.add(Activation('relu'))
        model_net.add(Dropout(0.5))
        model_net.add(Dense(3))
        model_net.add(Activation('sigmoid'))

        model_net.compile(loss='binary_crossentropy',
                          optimizer=sgd,
                          metrics=['accuracy'])
        model_net.fit_generator(train_gen, steps_per_epoch=num_train_samples // batch_size,
                                epochs=epochs, validation_data=val_gen, validation_steps=num_val_samples)
        model_net.save(save_model_name, include_optimizer=False)
        scores = model_net.evaluate_generator(test_gen, num_test_samples)

        print('Точность на тестовых данных составляет: %.2f%%' % (scores[1] * 100))


model = WorkWithModel()
# model.create_test_val_train_data('DataSet1', ['frolov', 'khudyakov', 'semin'])
path2train_dir = 'DataSet\\train'
path2test_dir = 'DataSet\\test'
path2val_dir = 'DataSet\\val'
model.train_network(path2train_dir, path2test_dir, path2val_dir, 3, 17, 1e-4, 'test.h5')
