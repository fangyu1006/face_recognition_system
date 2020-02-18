import tensorflow as tf
with tf.Session() as sess:
    with open('/home/iim/lin/dan/ego/landmark_dan_tf/models/dan.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read()) 
        print graph_def
