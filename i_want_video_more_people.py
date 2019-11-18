import numpy as np
import cv2 as cv
from mtcnn.mtcnn import MTCNN

max_num_people = 20
#class GiveMeVideo:
#    def __init__(self):
cap = cv.VideoCapture(0)
detector = MTCNN()

while True:
    count_frame = 0
    ret, frame = cap.read()
    frame2 = frame.copy()
    face = detector.detect_faces(frame)
    num_people = 0
    while len(face):
        point1 = (face[0]['box'][0], face[0]['box'][1])
        point2 = (face[0]['box'][0] + face[0]['box'][2], face[0]['box'][1] + face[0]['box'][3])
        cv.rectangle(frame, point1, point2, (0, 0, 255), 3)
        frame2[point1[1]:point2[1], point1[0]:point2[0], :] = 0
        face = detector.detect_faces(frame2)
        num_people += 1
        if num_people > max_num_people:
            break
    print(num_people)
    cv.imshow('Video', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()


#video = GiveMeVideo()
