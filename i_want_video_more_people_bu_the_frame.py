import numpy as np
import cv2 as cv
import imutils as im
from mtcnn.mtcnn import MTCNN

max_num_people = 20
num_frames_skip = 4
cap = cv.VideoCapture(0)
detector = MTCNN()
count_frame = 0
while True:
    lst = []
    if count_frame == num_frames_skip:
        count_frame = 0
    ret, frame = cap.read()
    if not count_frame:
        points_face = []
        eyes_coor = []
        frame2 = frame.copy()
        face = detector.detect_faces(frame)
        num_people = 0
        while len(face):
            points_face.append((face[0]['box'][0], face[0]['box'][1]))
            points_face.append((face[0]['box'][0] + face[0]['box'][2], face[0]['box'][1] + face[0]['box'][3]))
            #cv.rectangle(frame, points_face[num_people], points_face[num_people + 1], (0, 0, 255), 3)
            frame2[points_face[num_people][1]:points_face[num_people + 1][1],
                   points_face[num_people][0]:points_face[num_people + 1][0], :] = 0
            lst.append(frame[points_face[num_people][1]:points_face[num_people + 1][1],
                             points_face[num_people][0]:points_face[num_people + 1][0], :])

            eyes_coor.append(face[0]['keypoints']['right_eye'])
            eyes_coor.append(face[0]['keypoints']['left_eye'])
            face = detector.detect_faces(frame2)
            num_people += 1
            if num_people > max_num_people:
                break
        print(num_people)
        count_frame += 1
    else:
        for i in range(0, 2*num_people, 2):
            cv.rectangle(frame, points_face[i], points_face[i + 1], (0, 0, 255), 3)
            cv.circle(frame, eyes_coor[i], 2, (255, 0, 0), 2)
            cv.circle(frame, eyes_coor[i+1], 2, (255, 0, 0), 2)
        count_frame += 1
    cv.imshow('Video', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
