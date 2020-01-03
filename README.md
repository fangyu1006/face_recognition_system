# face_recognition_system

## Introduction
This project is for face recognition demo.  The face recognition system mainly inlcudes face detection, landmarks estimation, face alignment and face identification. [MTCNN](https://kpzhang93.github.io/MTCNN_face_detection_alignment/) is used for face detection and landmarks estimation. For face identification, [Insightface](https://github.com/deepinsight/insightface) are recommended.

## Usage
+ Prepare a recognition model. You may download a pre-trained model from insightface [model-zoo](https://github.com/deepinsight/insightface/wiki/Model-Zoo).
+ Prepare face image data set as reference images and aligned to 112x112.
+ Extract feartures of reference images using register_xx.py. xx is the framework the model use.
```Bash
python register_mx.py
```
+ Modify the run.sh and run it.
```Bash
./run.sh
```
