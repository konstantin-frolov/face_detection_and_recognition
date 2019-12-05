# Face detection and recognition  
This project was created to implement simple face recognition by video from a webcam.  
In this case used:
* Python 3.6.7
* Tensorflow 2.0.0
* Keras 2.3.1
* Numpy 1.17.4
* OpenCV contrib 4.1.1.26
* MTCNN (I don’t know why, but the official version does not work with TF2.0, in my [repo](https://github.com/konstantin-frolov/mtcnn) you can download a working solution for MTCNN)
* FaceNet model  
There are two base classes throughout the work: GiveMevideo in i_want_video.py and WorkWithModel in work_with_model.py  

## class GiveMeVideo  
In this class we have 6 public and 3 private methods for working with webcam video.

### Public methods:
#### ``GiveMeVideo.only_video()``
This method has no inputs and outputs. Use this method to display video from a webcam without processing in an endless loop. Click key "Q" to exit from loop.
#### ``GiveMeVideo.detect_1_face()``
This method has no inputs and outputs. Use this method to detect only one face in a video frame. Face recognition is done using MTCNN and a red rectangle is drawn around the face when showing a video.

#### ``GiveMeVideo.detect_more_faces(max_num_people, num_frames_skip)``
This method has 2 inputs and no outputs.  
inputs:  
* ``max_num_people`` - maximum number of faces to detect;  
* ``num_frames_skip`` - number of skipped frames for faster work.  

Why is this needed? Because every frame from the video goes to MTCNN to detect one face.
Then, the image area with the detected face changes to black throughout the frame and returns to MTCNN to detect a new face.
This cycle works until all faces are detected or max_num_people is reached.
If you change num_frames_skip, MTCCN will detect all faces in one frame and red rectangles will be drawn around all faces in all skipped frames.
If you set the num_frames_skip to a large value, you can see freezing when drawing a rectangle around the face between frames.

#### ``GiveMeVideo.get_only_face()``
Method has no inputs and outputs. Use this method if you want show only detected face.
In it used private method ``__get_one frame``. Info about it below.  
#### ``GiveMeVideo.create_face_dataset(user_name, folder_path, num_files)``  
This method has 3 inputs and no outputs. Use this method to create dataset of face images for training your neuronetwork.
A single use of the method allows you to create data for one user that the system recognizes.  
inputs:  
* ``user_name`` - recognition user name, used to create folders and then write to the image after recognition;
* ``folder_path`` - path to folder where you can save images. Default value - DataSet\\all_data;
* ``num_files`` - number of files for training neural network. The more files, the better the neural network will learn. Default value - 1000.  

After using this method for all recognition users, you have folders with images of user faces.
#### ``GiveMeVideo.get_video_recognition(model_name)``  
This method has 1 input and no outputs. Use this method for show video after detecting and recognition only one face using Keras model in *h5 file.  
input:
* ``model_name`` - path to your trained model for recognition faces in format ``path\\to\\you\\model\\model_name.h5``
### Private methods:
#### ``GiveMeVideo.__get_one_frame``  
Method has no inputs and 3 outputs. It is used to detect the face in the frame, if it is detected, crop the face from the image, add the coordinates of the face in the tuple (upper left point and lower right corner) and change the face detection mark to True.
outputs:
* ``face_img`` - only face into image;
* ``face_points`` - the coordinates of the face in the tuple (upper left point and lower right corner). If face don't detect is ``None``;
* ``face_detection_mark`` - If face detect is ``True``. Else - ``False``  

#### ``GiveMeVideo.__find_face(frame)``  
Method has 1 input and 2 outputs. This methods works like ``__get_one_frame`` without initializing the camera. You need to give an image frame to the input of this method.  
input:
* ``frame`` - frame image from camera.  

outputs:
* ``face_points`` - the coordinates of the face in the tuple (upper left point and lower right corner). If face don't detect is ``False``;
* ``face_img`` - only face into image. If face don't detect is ``False``.  

#### ``GiveMeVideo.__face_recognition(face_img, model)``  
Method has 2 inputs and 1 output. It is used to recognize face from face_img and return the index of the maximum element in the prediction list.  
inputs:
* ``face_img`` - only face into image;  
* ``model`` - object of your loaded and compiled Keras model.  

outputs:
* ``pred_index`` - index of the maximum element in the prediction list.  

## class WorkWithModel  
Information in developing
