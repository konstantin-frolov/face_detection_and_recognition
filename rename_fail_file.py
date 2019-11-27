import os
import shutil
import numpy as np

def rename_files_in_dir(dir_name, files_name, new_files_name):
    for i in range(1, len(os.listdir(dir_name))):
        #os.rename(dir_name + os.listdir(dir_name)[i-1], dir_name + new_files_name + str(1000+i) + '.jpg')
        os.rename(dir_name+files_name+str(i)+'.jpg', dir_name+new_files_name+str(i+899)+'.jpg')


def rename_all_files(dir_name, new_files_name):
    i = 1
    for filename in os.listdir(dir_name):
        new_names = dir_name + new_files_name + str(i+899) + '.jpg'
        src = dir_name+filename
        os.rename(src, new_names)
        i += 1


def create_test_val_train_data(dir_name, class_names):
    train_dir = 'train'
    val_dir = 'val'
    test_dir = 'test'
    test_portion = 0.1
    val_portion = 0.1
    create_dirs(os.path.join(dir_name, train_dir))
    create_dirs(os.path.join(dir_name, test_dir))
    create_dirs(os.path.join(dir_name, val_dir))
    for i in range(len(class_names)):
        create_dirs(os.path.join(dir_name, train_dir, class_names[i]))
        create_dirs(os.path.join(dir_name, test_dir, class_names[i]))
        create_dirs(os.path.join(dir_name, val_dir, class_names[i]))
        nb_imgs = len(os.listdir(os.path.join(dir_name, 'all_data', class_names[i])))
        index_list = list(range(1, nb_imgs+1))
        np.random.shuffle(index_list)
        copy_imgs(0, test_portion*nb_imgs, index_list, os.path.join(dir_name, 'all_data', class_names[i]),
                  os.path.join(dir_name, test_dir, class_names[i]), class_names[i])
        copy_imgs(test_portion * nb_imgs + 1, test_portion * nb_imgs + 1 + val_portion*nb_imgs,
                  index_list, os.path.join(dir_name, 'all_data', class_names[i]),
                  os.path.join(dir_name, val_dir, class_names[i]), class_names[i])
        copy_imgs(test_portion * nb_imgs + 1 + val_portion * nb_imgs, nb_imgs,
                  index_list, os.path.join(dir_name, 'all_data', class_names[i]),
                  os.path.join(dir_name, train_dir, class_names[i]), class_names[i])


def create_dirs(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.mkdir(dir_name)
    # os.makedirs(os.path.join(dir_name, class_name))


def copy_imgs(start_index, end_index, index_list, source_dir, dest_dir, class_name):
    for i in range(int(start_index), int(end_index)):
        shutil.copy2(os.path.join(source_dir, class_name + '.' + str(index_list[i]) + '.jpg'), dest_dir)


create_test_val_train_data('DataSet', ['frolov', 'khudyakov', 'semin'])

#rename_all_files('C:\\Users\\frolov\\PycharmProjects\\face_detection_and_recognition\\DataSet\\test\\semin\\', 'semin.')
