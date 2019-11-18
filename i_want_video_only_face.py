import numpy as np
import cv2 as cv
from mtcnn.mtcnn import MTCNN
import imutils as im

max_num_people = 20
#class GiveMeVideo:
#    def __init__(self):
cap = cv.VideoCapture(0)
detector = MTCNN()

while True:
    eyes_coor = []
    ret, frame = cap.read()
    frame2 = frame.copy()
    face = detector.detect_faces(frame)
    num_people = 0
    if len(face):
        point1 = (face[0]['box'][0], face[0]['box'][1])
        point2 = (face[0]['box'][0] + face[0]['box'][2], face[0]['box'][1] + face[0]['box'][3])
        #cv.rectangle(frame, point1, point2, (0, 0, 255), 3)
        frame = frame[point1[1]-10:point2[1]+10, point1[0]-10:point2[0]+10, :]
        eyes_coor.append(face[0]['keypoints']['right_eye'])
        eyes_coor.append(face[0]['keypoints']['left_eye'])
        angle = -180/np.pi*np.arctan((eyes_coor[0][1]-eyes_coor[1][1])/(eyes_coor[0][0]-eyes_coor[1][0]))
        frame = im.rotate_bound(frame, angle)
        print(angle)
        #cv.circle(frame, eyes_coor[0], 1, (0, 255, 255), 2)
        #cv.circle(frame, eyes_coor[1], 1, (0, 255, 255), 2)
    cv.imshow('Video', frame)
    cv.imshow('Video', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()


#video = GiveMeVideo()
