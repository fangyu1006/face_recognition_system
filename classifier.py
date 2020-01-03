import tensorflow as tf
import numpy as np
import os
from glob import glob
import mxnet as mx
import cv2

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


class Resnet20:
    def __init__(self, model_path, ref_path, model_type):
        self.model_type = model_type
        if self.model_type == "tf":
            self.net, self.inputs, self.prelogits = load_graph_tf(model_path,"prefix/REC/INPUT:0","prefix/REC/OUTPUT:0")
            #self.net, self.inputs, self.prelogits = load_graph_tf(model_path,"prefix/data:0","prefix/fc1/add_1:0")
        self.person_names = self.map_labels(ref_path)
        self.ref_features, self.ref_labels = self.read_ref_features(ref_path)
        if model_type == "mxnet":
            self.model = get_model(model_path)
        print("init...")

    def get_feature(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images_patch = []
        images_patch.append(img)
        embeddings = extract_feature_batch(np.asarray(images_patch), self.net, self.inputs, self.prelogits)
        feature = np.reshape(np.asarray(embeddings[0]),[1,-1]).flatten()
        feature = feature/np.linalg.norm(feature)
        return feature
    
    def get_feature_mx(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2,0,1))
        input_blob = np.expand_dims(img, axis=0)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        self.model.forward(db, is_train=False)
        embedding = self.model.get_outputs()[0].asnumpy()
        embedding = embedding.flatten()
        embedding = embedding/np.linalg.norm(embedding)
        return embedding

    def get_features(self, imgs):
        images_patch = []
        for img in imgs:
            images_patch.append(img)
        embeddings = extract_feature_batch(np.asarray(images_patch), self.net, self.inputs, self.prelogits)
        features = []
        for feature in embeddings:
            feature = feature/np.linalg.norm(feature)
            features.append(feature)
        return np.array(features)


    def map_labels(self, ref_path):
        person_names = []
        for person_dir in glob(os.path.join(ref_path, "*", "")):
            person = person_dir.split('/')[-2]
            person_names.append(person)
        return np.asarray(person_names)
    
    def get_label(self, name):
        label = np.where(self.person_names == name)
        return label[0][0]



    def read_ref_features(self, ref_path):
        features = []
        labels = []
        for root, dirs, files in os.walk(ref_path):
            for name in files:
                if name.endswith('.txt'):
                    txt_path = os.path.join(root, name)
                    person_name = txt_path.split('/')[-2]
                    label = self.get_label(person_name)
                    labels.append(label)
                    with open(txt_path, 'r') as fd:
                        f = fd.readline()
                        feature = np.array([float(i) for i in f.split(' ')[0:512]])
                        features.append(feature)
                                            
        return np.array(features), np.array(labels)
                    



    def predict(self, img):
        feature = self.get_feature(img)
        feature_dists = feature.dot(self.ref_features.T)
        pred_idx = np.argmax(feature_dists)
        pred = self.ref_labels[pred_idx]
        return pred
    
    def predict_batch(self, imgs):
        preds = []
        features = self.get_features(imgs)

        feature_dists = features.dot(self.ref_features.T)
        preds_idx = np.argmax(feature_dists, axis=1)
        for idx in preds_idx:
            pred = self.ref_labels[idx]
            preds.append(pred)

        return np.array(preds)

    def get_recog_result(self, img, threshold):
        threshold = threshold*2-1
        if self.model_type == "mxnet":
            feature = self.get_feature_mx(img)
        else:
            feature = self.get_feature(img)
        feature_dists = feature.dot(self.ref_features.T)
        pred_idx = np.argmax(feature_dists)
        #print(feature_dists[pred_idx])
        if feature_dists[pred_idx] > threshold:
            pred = self.ref_labels[pred_idx]
            return self.person_names[pred], feature_dists[pred_idx]
        return None,0
    

