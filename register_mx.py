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
import mxnet as mx

def get_model(model_str):
    ctx = mx.gpu(0)
    _vec = model_str.split(',')
    assert len(_vec)==2
    prefix = _vec[0]
    epoch = int(_vec[1])
    print('loading',prefix, epoch)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    all_layers = sym.get_internals()
    sym = all_layers['fc1_output']
    model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
    model.bind(data_shapes=[('data', (1, 3, 112, 112))])
    model.set_params(arg_params, aux_params)
    return model


def extract_feature_mx(model,img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2,0,1))
    #img = (img - 127.5)*0.0078125
    input_blob = np.expand_dims(img, axis=0)
    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=(data,))
    model.forward(db, is_train=False)
    embedding = model.get_outputs()[0].asnumpy()
    embedding = embedding.flatten()
    embedding = embedding/np.linalg.norm(embedding)
    #embedding = sklearn.preprocessing.normalize(embedding)i
    return embedding


def get_features(model, src_path, dst_path):
    for root, dirs, files in os.walk(src_path):
        for name in files:
            img_path = os.path.join(root, name)
            feature = extract_feature_mx(model, img_path)
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
    #model_str = "/home/fangyu/git/insightface/models/r18-arcface-emotion/model,1" 
    model_str = "/home/fangyu/fy/face-recognition-benchmarks/IIM/models/v1-arcface-emotion-kd/model,1" 
    dst_path = "./ref_features/v1-arcface-emotion-kd/"
    src_path = "/home/fangyu/fy/face-recognition-benchmarks/IIM/iim_dataset_registration-4827/dataset_112x112/"

    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    model = get_model(model_str)
    get_features(model, src_path, dst_path)

