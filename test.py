from i_want_video import GiveMeVideo


vid = GiveMeVideo()
vid.get_video_recognition_skip_frames('models_weights\\face_recognition_ep=7_facenet_with_conv.h5', 4)
#vid.get_only_face()
#vid.create_face_dataset('DataSet\\test\\chelovek1\\', 'chelovek1')

#vid.get_only_face()

