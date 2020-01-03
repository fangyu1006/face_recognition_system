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
import face_preprocess
from utils import draw_txt_image
import argparse

reload(sys)
sys.setdefaultencoding('utf8')


def parse_args():
    parser = argparse.ArgumentParser(description="face recognition system")
    parser.add_argument('--video_path', type=str, default='./test.avi', help='tested video path')
    parser.add_argument('--ref_path', type=str, default='./ref_features/r18-arcface-emore/', help='registered images feature path')
    parser.add_argument('--model_path', type=str, default='./models/r18-arcface-emore/model,1', help='tested model path')
    parser.add_argument('--model_type', type=str, default='mxnet', help='model type, mxnet or tf')
    parser.add_argument('--threshold', type=float, default=0.75, help='score threshold')
    parser.add_argument('--save_video', type=bool, default=False, help='whether to save output video')
    parser.add_argument('--save_path', type=str, default='./videos/output.avi', help='out video save path')

    args = parser.parse_args()
    return args


args = parse_args()
# Detection settings
det_minsize = 90
det_threshold = [0.6,0.7,0.8]
mtcnn_path = "./mtcnn-model/"
# Alignment settings
img_size = '112,112'

true_name = "章泉_2442章泉_2444章泉_2443章泉_2441"
#true_name = "Yue_Jiaxin"

# Initilaize
ctx = mx.gpu(0)
detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, minsize=det_minsize,
                        accurate_landmark=True, threshold=det_threshold)
recog_classifier = Resnet20(args.model_path, args.ref_path, args.model_type)
video_capture = cv2.VideoCapture(args.video_path)
det_cnt = 0
recog_cnt = 0
error = 0


if args.save_video:
    frame_width = int(video_capture.get(3))
    frame_height = int(video_capture.get(4))
    out = cv2.VideoWriter(args.save_path,cv2.VideoWriter_fourcc(*'XVID'), 10, (frame_width,frame_height))
while True:
    ret, frame = video_capture.read()

    if ret:
        det_results = detector.detect_face(frame)
        if det_results is not None:
            bboxes, points = det_results
            if bboxes.shape[0] == 0:
                continue
            #print(bboxes)
            
            #print(points)
            for i in range(bboxes.shape[0]):
                det_cnt += 1
                bbox = bboxes[i, :4]
                score = bboxes[i, 4]
                landmark = points[i].reshape((2,5)).T
                face = face_preprocess.preprocess(frame, bbox, landmark, image_size=img_size)
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                recog_ret, score = recog_classifier.get_recog_result(face,args.threshold)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                            (255,0,0), 2)
                #if recog_ret is not None and recog_ret == "Yue_Jiaxin":
                if recog_ret is not None:
                    recog_cnt += 1
                    if recog_ret != true_name:
                        error += 1
                    frame = draw_txt_image(frame, recog_ret, (int(bbox[0]), int(bbox[1])-50), 40, (0,255,255))
                    #cv2.putText(frame, unicode(recog_ret), (int(bbox[0]), int(bbox[1])-4), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)
                    cv2.putText(frame,str((score+1)/2), (int(bbox[2]), int(bbox[3])-18), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)
        cv2.putText(frame, "recog rate: " + '{:d}'.format(recog_cnt) + "/" + '{:d}'.format(det_cnt), (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2) 
        cv2.putText(frame, "error rate: " + '{:d}'.format(error) + "/" + '{:d}'.format(recog_cnt), (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2) 
            #for i in range(points.shape[0]):
                #landmark = points[i]
                #for j in range(len(landmark)/2):
                    #cv2.circle(frame, (int(landmark[j]), int(landmark[j+5])), 5, (0,255,255))
        
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

