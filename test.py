from i_want_video import GiveMeVideo


vid = GiveMeVideo()
vid.get_video_recognition_skip_frames('models_weights\\face_recognition_ep=7_facenet_with_conv.h5', 24)
#vid.create_face_dataset('DataSet\\test\\chelovek1\\', 'chelovek1')

#vid.get_only_face()

