import numpy as np
import cv2 as cv
from mtcnn.mtcnn import MTCNN
from keras.preprocessing import image
from keras.models import load_model
from os import listdir


def rotate_img(img, angle, center):
    row, col, num_colors = img.shape
    rot_mat = cv.getRotationMatrix2D(center, angle, 1.0)
    img = cv.warpAffine(img, rot_mat, (col, row))
    return img


class GiveMeVideo:
    def __init__(self):
        self.cap = cv.VideoCapture(0)
        self.detector = MTCNN()

    def only_video(self):
        while True:
            ret, frame = self.cap.read()
            cv.imshow('Video', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv.destroyAllWindows()

    def detect_1_face(self):
        while True:
            ret, frame = self.cap.read()
            face = self.detector.detect_faces(frame)
            if len(face):
                point1 = (face[0]['box'][0], face[0]['box'][1])
                point2 = (face[0]['box'][0] + face[0]['box'][2], face[0]['box'][1] + face[0]['box'][3])
                cv.rectangle(frame, point1, point2, (0, 0, 255), 3)

            cv.imshow('Video', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv.destroyAllWindows()

    def detect_more_faces(self, max_num_people, num_frames_skip):
        count_frame = 0
        while True:
            lst = []
            if count_frame == num_frames_skip:
                count_frame = 0
            ret, frame = self.cap.read()
            if not count_frame:
                points_face = []
                eyes_coor = []
                frame2 = frame.copy()
                face = self.detector.detect_faces(frame)
                num_people = 0
                while len(face):
                    points_face.append((face[0]['box'][0], face[0]['box'][1]))
                    points_face.append((face[0]['box'][0] + face[0]['box'][2], face[0]['box'][1] + face[0]['box'][3]))
                    frame2[points_face[num_people][1]:points_face[num_people + 1][1],
                    points_face[num_people][0]:points_face[num_people + 1][0], :] = 0
                    lst.append(frame[points_face[num_people][1]:points_face[num_people + 1][1],
                               points_face[num_people][0]:points_face[num_people + 1][0], :])

                    eyes_coor.append(face[0]['keypoints']['right_eye'])
                    eyes_coor.append(face[0]['keypoints']['left_eye'])
                    face = self.detector.detect_faces(frame2)
                    num_people += 1
                    if num_people > max_num_people:
                        break
                count_frame += 1
            else:
                for i in range(0, 2 * num_people, 2):
                    cv.rectangle(frame, points_face[i], points_face[i + 1], (0, 0, 255), 3)
                    cv.circle(frame, eyes_coor[i], 2, (255, 0, 0), 2)
                    cv.circle(frame, eyes_coor[i + 1], 2, (255, 0, 0), 2)
                count_frame += 1
            cv.imshow('Video', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv.destroyAllWindows()

    # Method
    def get_one_frame(self):
        eyes_coor = []
        ret, frame = self.cap.read()
        face = self.detector.detect_faces(frame)
        if len(face):
            # Дернем координаты глаз и рассчитаем угол
            eyes_coor.append(face[0]['keypoints']['right_eye'])
            eyes_coor.append(face[0]['keypoints']['left_eye'])
            angle = 180 / np.pi * np.arctan(
                (eyes_coor[0][1] - eyes_coor[1][1]) / (eyes_coor[0][0] - eyes_coor[1][0]))
            # Повернем изображение, чтобы линия глаз была горизонтальной
            frame = rotate_img(frame, angle, face[0]['keypoints']['nose'])
            # Выдернем только лицо со всей картинки
            point1 = (face[0]['box'][0], face[0]['box'][1])
            point2 = (face[0]['box'][0] + face[0]['box'][2], face[0]['box'][1] + face[0]['box'][3])
            face_points = [point1, point2]
            frame = frame[point1[1] - 10:point2[1] + 10, point1[0] - 10:point2[0] + 10, :]
            # масштабируем под размер 300 px
            new_width = 224
            width = abs(point1[0] - point2[0])
            height = abs(point1[1] - point2[1])
            new_height = int(new_width * width / height)
            frame = cv.resize(frame, (new_height, new_width))
            return frame, face_points, 1
        else:
            return frame, None, 0

    # Method to show only detecting face
    # inputs: nothing
    # outputs: nothing
    def get_only_face(self):
        while True:
            frame, mark_face = GiveMeVideo.get_one_frame(self)
            cv.imshow('Video', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv.destroyAllWindows()

    # Method for creating dataset for training
    # inputs: user_name - name user for training to recognize
    #         folder_path - path to folder for all images, default - DataSet\all_data
    #         num_files - numbers of images for training, validating and testing, default = 1000
    # output: only images in folder and value i in process
    def create_face_dataset(self, user_name, folder_path='DataSet\\all_data', num_files=1000):
        i = 1
        while i < num_files:
            frame, mark_face = GiveMeVideo.get_one_frame(self)
            if mark_face:
                path_to_img = folder_path + user_name + '.' + str(i) + '.jpg'
                cv.imwrite(path_to_img, frame)
                print(iter)
                i += 1

    # Method for getting one face_img for recognizing and face points from one frame using MTCCN
    # inputs: frame - one frame from camera
    # output: face_points - list from 2 points top left and bottom right of face
    def find_face(self, frame):
        eyes_coor = []
        face = self.detector.detect_faces(frame)
        if len(face):
            eyes_coor.append(face[0]['keypoints']['right_eye'])
            eyes_coor.append(face[0]['keypoints']['left_eye'])
            angle = 180 / np.pi * np.arctan(
                (eyes_coor[0][1] - eyes_coor[1][1]) / (eyes_coor[0][0] - eyes_coor[1][0]))
            frame = rotate_img(frame, angle, face[0]['keypoints']['nose'])

            point1 = (face[0]['box'][0], face[0]['box'][1])   # top left point
            point2 = (face[0]['box'][0] + face[0]['box'][2],
                      face[0]['box'][1] + face[0]['box'][3])  # bottom right point
            face_points = [point1, point2]                    # two point in list
            face_img = frame[point1[1]:point2[1], point1[0]:point2[0], :]  # get face_img from frame by face points
            return face_points, face_img  # return face points and face_img or False
        else:
            return False, False

    # Method for recognizing face
    # inputs: face_img from find_face, model - your trained model with weights
    # output: index of max(preds) + 1 or 0
    @staticmethod
    def face_recognition(face_img, model):
        face_img = cv.resize(face_img, (160, 160))  # resize face_img for model inputs size
        img_array = image.img_to_array(face_img)    # convert to array, add new axis and scale to 0:1 values
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.
        preds = model.predict(img_array)            # put face_img to inputs of your model and get prediction
        # get top max from prediction if it more than 0.5
        if np.max(preds) > 0.5:
            return np.argmax(preds) + 1  # return with bias of 1 for correct interpretation of answer list
        else:
            return 0

    # Method to get frame from cam and recognize one face in it
    # inputs: model_name - model with weights in *.h5 file
    # output: nothing
    def get_video_recognition(self, model_name):
        # load model
        model = load_model(model_name, compile=False)
        # Create answer list
        names = ["I don't know"]
        names.extend(listdir('DataSet\\all_data'))
        # inf loop for getting frame from camera
        while True:
            ret, frame = self.cap.read()
            face_rext_points, face_img = GiveMeVideo.find_face(self, frame)  # Getting points of the face
            if face_rext_points:  # If we have face points
                cv.rectangle(frame, face_rext_points[0], face_rext_points[1],
                             (0, 0, 255), 3)                                 # rectangle around the face
                name = GiveMeVideo.face_recognition(face_img, model)   # Recognize face
                cv.putText(frame, names[name], face_rext_points[1],
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
                           cv.LINE_AA)                                       # Simple text to the right of the face
            cv.imshow('Video', frame)                                        # Show processed frame

            # If click on key "Q" - exit
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv.destroyAllWindows()




