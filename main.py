#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import os
import sys
import cv2
import numpy as np
import mxnet as mx
import tensorflow as tf

from mtcnn_detector import MtcnnDetector
from classifier import Resnet20
from landmark_dan import landmarkDan
import face_preprocess
from utils import draw_txt_image
import argparse
from retinaface_cov import RetinaFaceCoV

reload(sys)
sys.setdefaultencoding('utf8')


def parse_args():
    parser = argparse.ArgumentParser(description="face recognition system")
    parser.add_argument('--video_path', type=str, default='./test.avi', help='tested video path')
    parser.add_argument('--ref_path', type=str, default='./ref_features/r18-arcface-deepglint32/', help='registered images feature path')
    parser.add_argument('--model_path', type=str, default='./models/r18-arcface-deepglint32/model,1', help='tested model path')
    parser.add_argument('--model_type', type=str, default='mxnet', help='model type, mxnet or tf')
    parser.add_argument('--threshold', type=float, default=0.75, help='score threshold')
    parser.add_argument('--save_video', type=bool, default=False, help='whether to save output video')
    parser.add_argument('--save_path', type=str, default='./videos/output.mp4', help='out video save path')
    parser.add_argument('--detector', type=str, default='MTCNN', help='type of face detector, MTCNN of Retinaface')
    parser.add_argument('--landmark', type=str, default='MTCNN', help='type of landmark estimation, MTCNN of DAN')

    args = parser.parse_args()
    return args


args = parse_args()

# Detection settings
if args.detector == "MTCNN":
    det_minsize = 90
    det_threshold = [0.6,0.7,0.8]
    mtcnn_path = "./mtcnn-model/"
elif args.detector == "Retinaface":
    face_threshold = 0.8
    mask_threshod = 0.2
    #scales = [640, 1080]
    scales = [160, 270]
    target_size = scales[0]
    max_size = scales[1]

# Alignment settings
img_size = '112,112'

true_name = "章泉_2442章泉_2444章泉_2443章泉_2441"
#true_name = "Yue_Jiaxin"

# Initilaize
gpuid = 0
ctx = mx.gpu(gpuid)


if args.video_path is "0":
    video_capture = cv2.VideoCapture(0)
else:
    video_capture = cv2.VideoCapture(args.video_path)
frame_width = int(video_capture.get(3))
frame_height = int(video_capture.get(4))

if args.detector == "MTCNN":
    detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, minsize=det_minsize,
                        accurate_landmark=True, threshold=det_threshold)
elif args.detector == "Retinaface":
    detector = RetinaFaceCoV('./retinaface_model/mnet_cov2', 0, gpuid, 'net3l')
    im_size_min = np.min((frame_width, frame_height))
    im_size_max = np.max((frame_width, frame_height))
    im_scale = float(target_size) / float(im_size_min)
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    print('im_scale', im_scale)

if args.landmark is "DAN":
    landmarkEstimator = landmarkDan("./dan_models/dan.pb", 0.9, "eyeNoseLips")
recog_classifier = Resnet20(args.model_path, args.ref_path, args.model_type)

det_cnt = 0
recog_cnt = 0
error = 0
rectangle_color = (0,255,0)

if args.save_video:
    out = cv2.VideoWriter(args.save_path,cv2.VideoWriter_fourcc(*'XVID'), 10, (frame_width,frame_height))
while True:
    ret, frame = video_capture.read()



    if ret:
        if args.detector == "MTCNN":
            det_results = detector.detect_face(frame)
            if det_results is None:
                continue
            else:
                bboxes, points = detector.detect_face(frame)    
        elif args.detector == "Retinaface":
            scales = [0.5,0.5]
            bboxes, points = detector.detect(frame, face_threshold, scales=scales, do_flip=False)

        if bboxes is not None:
            #bboxes, points = det_results
            if bboxes.shape[0] == 0:
                continue
            #print(bboxes)
            
            #print(points.shape)
            if args.landmark is "DAN":
                img_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for i in range(bboxes.shape[0]):
                det_cnt += 1
                bbox = bboxes[i, :4]
                score = bboxes[i, 4]
                if args.detector == "Retinaface":
                    mask = bboxes[i, 5]
                    if mask >= mask_threshod:
                        rectangle_color = (0,0,255)
                    else:
                        rectangle_color = (0,255,0)

                if args.landmark is "DAN":
                    landmark, l_flag = landmarkEstimator.get_landmarks(img_bw, bbox)
                    if l_flag is False:
                        continue
                else:
                    #landmark = points[i].reshape((2,5)).T
                    landmark = points[i]
                    if landmark.shape[0] != 5:
                        landmark = landmark.reshape((2,5)).T

                face = face_preprocess.preprocess(frame, bbox, landmark, image_size=img_size)
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                recog_ret, score = recog_classifier.get_recog_result(face,args.threshold)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                            rectangle_color, 2)
                #if recog_ret is not None and recog_ret == "Yue_Jiaxin":
                if recog_ret is not None:
                    recog_cnt += 1
                    if recog_ret != true_name:
                        error += 1
                    frame = draw_txt_image(frame, recog_ret, (int(bbox[0]), int(bbox[1])-50), 40, (0,255,255))
                    #cv2.putText(frame, unicode(recog_ret), (int(bbox[0]), int(bbox[1])-4), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)
                    cv2.putText(frame,str((score+1)/2), (int(bbox[2]), int(bbox[3])-18), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)
            #cv2.putText(frame, "recog rate: " + '{:d}'.format(recog_cnt) + "/" + '{:d}'.format(det_cnt), (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2) 
            #cv2.putText(frame, "error rate: " + '{:d}'.format(error) + "/" + '{:d}'.format(recog_cnt), (50, 100),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2) 
            
            for i in range(bboxes.shape[0]):
                landmark = points[i]
                if landmark.shape[0] != 5:
                    landmark = landmark.reshape((2,5)).T
                for j in range(len(landmark)):
                    cv2.circle(frame, (int(landmark[j][0]), int(landmark[j][1])), 5, (0,255,255))
            
        if args.save_video:
            out.write(frame)
        cv2.imshow("", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        print("device not found")
        break

video_capture.release()
if args.save_video:
    out.release()
cv2.destroyAllWindows()

