import os


def rename_files_in_dir(dir_name, files_name, new_files_name):
    for i in range(1, len(os.listdir(dir_name))):
        #os.rename(dir_name + os.listdir(dir_name)[i-1], dir_name + new_files_name + str(1000+i) + '.jpg')
        os.rename(dir_name+files_name+str(i)+'.jpg', dir_name+new_files_name+str(i)+'.jpg')


def rename_all_files(dir_name, new_files_name):
    i = 1
    for filename in os.listdir(dir_name):
        new_names = dir_name + new_files_name + str(i) + '.jpg'
        src = dir_name+filename
        os.rename(src, new_names)
        i += 1


rename_all_files('C:\\Users\\frolov\\PycharmProjects\\face_detection_and_recognition\\DataSet\\val\\khudyakov\\', 'khudyakov.')
