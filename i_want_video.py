import numpy as np
import cv2 as cv
from mtcnn.mtcnn import MTCNN


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
            frame = frame[point1[1] - 10:point2[1] + 10, point1[0] - 10:point2[0] + 10, :]
            # От масштабируем под размер 300 px
            new_width = 300
            width = abs(point1[0] - point2[0])
            height = abs(point1[1] - point2[1])
            new_height = int(new_width * width / height)
            frame = cv.resize(frame, (new_height, new_width))
            return frame, 1
        else:
            return frame, 0

    def get_only_face(self):
        while True:
            frame, mark_face = GiveMeVideo.get_one_frame(self)
            cv.imshow('Video', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv.destroyAllWindows()

    def create_face_dataset(self, folder_path, user_name):
        iter = 1
        while iter < 1000:
            frame, mark_face = GiveMeVideo.get_one_frame(self)
            path_to_img = folder_path + user_name + '.' + str(iter) + '.jpg'
            if mark_face:
                cv.imwrite(path_to_img, frame)
                print(iter)
                iter += 1



