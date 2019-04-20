import tensorflow as tf
from AlexNet import AlexNet_loss, AlexNet
from tqdm import tqdm
import os
import math
import pickle
import cv2
import numpy as np

# 训练集采用generator的方式产生
# 测试集采用yield的方式产生

LEARNINF_RATE = 1e-4                                 #学习率
EPOCHES = 10                                         #训练轮数
BATCH_SIZE = 128                                     #批量大小
N_CLASSES = 10                                       #标签的维度
DROPOUT = 0                                          #dropout概率
IMAGE_H = 224                                        #图片大小
IMAGE_W = 224

model_path = "model"
model_name = "cifar10.ckpt"

# 训练集更适合用yield的方式进行
# load test data from "CIFAR-10-test-label.pkl"
with open('CIFAR-10-test-label.pkl','rb') as f:
    test_data = pickle.load(f)
batches = math.ceil(len(test_data) / float(BATCH_SIZE))
##########################################################

def get_one_hot_label(labels,depth):
        '''
        把标签二值化  返回numpy.array类型
    
        args:
            labels：标签的集合
            depth：标签总共有多少类
        '''    
        m = np.zeros([len(labels),depth])
        for i in range(len(labels)):
            m[i][labels[i]] = 1
        return m

def data_generation(value,image_size='NONE',depth=10,one_hot = False):
        '''
        获取图片数据，以及标签数据 注意每张图片维度为 n_w x n_h x n_c
    
        args:
            value:由(x,y)元组组成的numpy.array类型
                x:图片路径
                y:对应标签
            image_size:图片大小 'NONE':不改变图片尺寸 
            one_hot：把标签二值化
            depth:数据类别个数
        '''
        #图片数据集合
        x_batch = []
        #图片对应的标签集合
        y_batch = []    
        #遍历每一张图片
        for image in value:      
            if image_size == 'NONE':
                x_batch.append(cv2.imread(image[0])/255)    #标准化0-1之间
            else:
                x_batch.append(cv2.resize(cv2.imread(image[0]),image_size)/255)
            y_batch.append(image[1])    
        
        if one_hot == True:
            #标签二值化
            y_batch = get_one_hot_label(y_batch,depth)
        return  np.array(x_batch,dtype=np.float32), np.array(y_batch,dtype=np.float32)

# 测试集采用yield的方式产生
def get_batch(dataset, batch_size):
    indexes = np.arange(len(dataset))
    while True:
        for idx in range(batches):
            batch_indexs = indexes[idx * batch_size:(idx + 1) * batch_size]
            batch = [test_data[k] for k in batch_indexs]
            X, y = data_generation(batch, image_size=(IMAGE_H, IMAGE_W), one_hot=True)
            yield (X, y)

# 创建placeholder
inputs_placeholder = tf.placeholder(dtype=tf.float32,shape=[None, IMAGE_H, IMAGE_W, 3], name="INPUTS")
# 理论上这里写tf.int32更合适,写tf.float32也行
labels_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, N_CLASSES], name="INPUTS")
keep_prob_placeholder = tf.placeholder(tf.float32,name='keep_prob')

alex_outputs = AlexNet(inputs_placeholder, N_CLASSES, keep_prob_placeholder)
cost, accuracy = AlexNet_loss(alex_outputs, labels_placeholder)
#创建Saver op用于保存和恢复所有变量
saver = tf.train.Saver()
model_file = tf.train.latest_checkpoint(model_path)
get_test_data = get_batch(test_data, BATCH_SIZE)
with tf.Session() as sess:
    print("加载模型")
    saver.restore(sess, model_file)
    acc_sum = 0.0
    for i in tqdm(range(batches)):
        X, y = get_test_data.__next__()
        acc = sess.run(accuracy, feed_dict={inputs_placeholder:X,
                                          labels_placeholder:y,
                                          keep_prob_placeholder:DROPOUT})
        acc_sum += acc
        print("epoche a%d:acc=%f,avg_acc=%f" % (i, acc, acc_sum / (i+1)))
    print("avg_acc=", acc_sum / batches)

