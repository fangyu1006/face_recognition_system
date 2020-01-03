# -*- coding: utf-8 -*-
#!/usr/bin/env python

import os
import numpy as np
import pickle
import glob
import pandas as pd
from os.path import  abspath
from imutils import paths
import cv2
import sklearn
import tensorflow as tf
from PIL import Image

def load_graph_tf(frozen_graph_filename, name_input_node,name_output_node):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        name_model = f.readline()
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def,name='prefix')

    input_tensor  = graph.get_tensor_by_name(name_input_node)
    output_tensor = graph.get_tensor_by_name(name_output_node)
    net = tf.Session(graph=graph)
    return net,input_tensor,output_tensor

def run_tensorflow_model(images,net,input_tensor,output_tensor):
    feed_dict  = {input_tensor:images}
    return net.run(output_tensor, feed_dict)

def extract_feature_batch(images, net, input_tensor, output_tensor):
    images = images.astype(float)
    # image normalize
    #images = (images-np.array([138.048458,110.2242922,96.731122944]))*0.0078125
    embeddings = run_tensorflow_model(images,net,input_tensor,output_tensor)
    return embeddings


def get_features(model_name, src_path, dst_path):
    net, inputs, prelogits = load_graph_tf(model_name,"prefix/REC/INPUT:0","prefix/REC/OUTPUT:0")
    #net, inputs, prelogits = load_graph_tf(model_name,"prefix/data:0","prefix/fc1/add_1:0")
    
    for root, dirs, files in os.walk(src_path):
        for name in files:
            img_path = os.path.join(root, name)
            print(img_path)
            images_patch = []
            img = np.array(Image.open(img_path))
            images_patch.append(img)
            embeddings = extract_feature_batch(np.asarray(images_patch), net, inputs, prelogits)
            feature = np.reshape(np.asarray(embeddings[0]),[1,-1]).flatten()
            feature = feature/np.linalg.norm(feature)
            #print(feature.shape)
            #feature = extract_feature_caffe(model, img_path)
            person_name = img_path.split('/')[-2]
            dst_dir = os.path.join(dst_path, person_name)
            if not os.path.exists(dst_dir):
                os.mkdir(dst_dir)
            dst_img_path = os.path.join(dst_path, person_name, name)
            dst_txt = os.path.splitext(dst_img_path)[0] + '.txt'
            with open(dst_txt, 'w') as fd:
                for feat in feature:
                    fd.write(str(feat) + ' ')





if __name__ == "__main__":
    #dst_path = "./features/r18_enhance_emotion"
    dst_path = "./ref_features/y3-enhance"
    model_path = "/home/fangyu/Documents/tag/model_output.pb"
    #model_path = "/home/fangyu/fy/face-recognition-benchmarks/IIM/models/y3-arcface-emotion/y3.pb"
    #src_path = "/home/fangyu/Downloads/yaoke/Wangyiyi_112x112"
    src_path = "/home/fangyu/fy/face-recognition-benchmarks/IIM/iim_dataset_registration-4827/dataset_112x112/"
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    get_features(model_path, src_path, dst_path)

