import cv2
import tensorflow as tf
import numpy as np

def load_graph_tf(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def,name='')

    input_tensor  = graph.get_tensor_by_name("data:0")
    landmarks_tensor = graph.get_tensor_by_name("s1_output/BiasAdd:0")
    score_tensor = graph.get_tensor_by_name("s1_confidence:0")
    net = tf.Session(graph=graph)
    return net,input_tensor,landmarks_tensor, score_tensor


class landmarkDan:
    def __init__(self, model_path, confidence_threshold_, interest_pts):
        self.init_landmarks = np.array([
                28.00000109,  41.52647628,
                28.18473504,  48.8934134 ,
                29.00594703,  56.22849206,
                30.5429435 ,  63.44273186,
                33.3968232 ,  70.1509803 ,
                37.81301038,  75.96647891,
                43.24753297,  80.81965695,
                49.22703732,  84.7477753 ,
                56.00000049,  85.90432565,
                62.77296366,  84.74777536,
                68.7524633 ,  80.81965707,
                74.18698594,  75.96647908,
                78.60317317,  70.15098051,
                81.45705532,  63.4427321 ,
                82.99405423,  56.22849232,
                83.81526628,  48.89341367,
                84.00000268,  41.52647655,
                33.20508337,  36.05814238,
                36.70989031,  32.88769496,
                41.6581876 ,  31.95748471,
                46.76025605,  32.70177554,
                51.53587615,  34.70101921,
                60.46412056,  34.70101925,
                65.23974543,  32.70177563,
                70.34181388,  31.95748485,
                75.29011116,  32.88769515,
                78.79491333,  36.05814259,
                56.0000007 ,  40.50158401,
                56.00000068,  45.29959295,
                56.00000066,  50.06187099,
                56.00000063,  54.97180982,
                50.36656807,  58.21004616,
                53.07869068,  59.19360572,
                56.00000061,  60.06598136,
                58.92131054,  59.19360574,
                61.63343316,  58.21004621,
                38.9034587 ,  41.06531965,
                41.91323112,  39.29239232,
                45.56029645,  39.34859723,
                48.73460205,  41.80837988,
                45.30799447,  42.44913183,
                41.68195906,  42.3938487 ,
                63.26539934,  41.80837995,
                66.43970496,  39.34859733,
                70.08676554,  39.29239245,
                73.09653795,  41.06531982,
                70.31803758,  42.39384884,
                66.69200216,  42.44913193,
                45.14172213,  67.10212727,
                49.14101717,  65.52928347,
                53.18444367,  64.65380041,
                56.00000058,  65.38025904,
                58.81555751,  64.65380044,
                62.858984  ,  65.52928354,
                66.85827902,  67.10212737,
                62.98245514,  70.94509019,
                59.06514636,  72.62448049,
                56.00000055,  72.94832027,
                52.93485474,  72.62448046,
                49.01754598,  70.94509013,
                46.82482808,  67.32302216,
                53.14207023,  67.05480287,
                56.00000058,  67.36606083,
                58.85793092,  67.05480289,
                65.17517307,  67.32302225,
                58.91110914,  69.000645  ,
                56.00000057,  69.34732024,
                53.088892  ,  69.00064497])

        self.mSquareSize = (112,112)
        self.mean_shape_width = 56.000001581036699
        self.mean_shape_height = 53.946840933296095
        self.mean_shape_center_x = 56.000001885325709
        self.mean_shape_center_y = 58.930905181645514
        self.mean_shape_mean_x = 56.0
        self.mean_shape_mean_y = 56.0
        self.confidence_threshold = confidence_threshold_
        
        mean_img_ = cv2.imread("./dan_models/mean.jpg", 0)
        mean_img_.astype(float)
        self.mean_img = mean_img_
        #mean_img_.convertTo(self.mean_img, cv2.CV_32FC1, 1.0)
        dev_img_ = cv2.imread("./dan_models/dev.jpg", 0)
        dev_img_.astype(float)
        self.dev_img = dev_img_
        #dev_img_.convertTo(self.dev_img, cv2.CV_32FC1, 1.0)
        self.nInterestPts = interest_pts
        self.net, self.input_tensor, self.landmarks_tensor, self.score_tensor = load_graph_tf(model_path)
    

    def bestFitRect(self, bbox):
        box_center = np.zeros((2))
        width = bbox[2] - bbox[0] + 1
        height = bbox[3] - bbox[1] + 1
        box_center[0] = bbox[0] + width/2.0
        box_center[1] = bbox[1] + height/2.0
        scale = (width/self.mean_shape_width + height/self.mean_shape_height)/2.0
        return box_center, scale

    def cropResizeRotate(self, img, box_center, scale):
        translate_x = box_center[0] - self.mean_shape_center_x * scale
        translate_y = box_center[1] - self.mean_shape_center_y * scale

        #warp_mat = {1/scale, 0, -translate_x/scale, 0, 1/scale, -translate_y/scale}
        warp_mat = np.array([1/scale, 0, -translate_x/scale, 0, 1/scale, -translate_y/scale])
        affine_mat = warp_mat.reshape(2,3)
        dst = np.zeros(self.mSquareSize, img.dtype)
        dst = cv2.warpAffine(img, affine_mat, self.mSquareSize)
        return dst, translate_x, translate_y

    def preprocess(self, face_img):
        face_img_f = face_img.astype(float)
        face_img_f = (face_img_f - self.mean_img)/self.dev_img
        return face_img_f

    def get_eyeNoseLips(self, raw_landmarks):
        VecLandmark = np.zeros((5,2))
        for dots_idx in range(36, 42):
            VecLandmark[0, 0] += raw_landmarks[dots_idx, 0]
            VecLandmark[0, 1] += raw_landmarks[dots_idx, 1]
        VecLandmark[0, 0] /= 6
        VecLandmark[0, 1] /= 6

        for dots_idx in range(42, 48):
            VecLandmark[1, 0] += raw_landmarks[dots_idx, 0]
            VecLandmark[1, 1] += raw_landmarks[dots_idx, 1]
        VecLandmark[1, 0] /= 6
        VecLandmark[1, 1] /= 6

        VecLandmark[2, 0] = raw_landmarks[30, 0]
        VecLandmark[2, 1] = raw_landmarks[30, 1]
        VecLandmark[3, 0] = raw_landmarks[48, 0]
        VecLandmark[3, 1] = raw_landmarks[48, 1]
        VecLandmark[4, 0] = raw_landmarks[54, 0]
        VecLandmark[4, 1] = raw_landmarks[54, 1]

        return VecLandmark

    def get_landmarks(self, img_bw, bbox):
        box_center, scale = self.bestFitRect(bbox)
        face_img, translate_x, translate_y = self.cropResizeRotate(img_bw, box_center, scale)
        input_img = self.preprocess(face_img)

        feed_dict  = {self.input_tensor:input_img.reshape((1,112,112,1))}
        landmarks_data, score_data = self.net.run((self.landmarks_tensor,  self.score_tensor), feed_dict)
        #print(landmarks_data)
        #print(score_data)
        flag = True
        if (score_data[0][1] > self.confidence_threshold):
            raw_landmarks = np.zeros((68, 2))
            for k in range(68):
                raw_landmarks[k, 0] = (self.init_landmarks[2*k] + landmarks_data[0][2*k])*scale+translate_x
                raw_landmarks[k, 1] = (self.init_landmarks[2*k+1] + landmarks_data[0][2*k+1])*scale+translate_y
            
            if (self.nInterestPts == "eyeNoseLips"):
                VecLandmarks = self.get_eyeNoseLips(raw_landmarks)
            else:
                VecLandmarks = raw_landmarks
        else:
            VecLandmarks = np.array([0])
            flag = False

        return VecLandmarks, flag   








