import os
import shutil
import numpy as np


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


model = WorkWithModel()
model.create_test_val_train_data('DataSet1', ['frolov', 'khudyakov', 'semin'])
