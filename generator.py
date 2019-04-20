import math
import cv2
import os
import pickle
import numpy as np

class BatchGenerator(object):

    def __init__(self, target_size, batch_size=1, shuffle=True):
        self.training_data = self.load_data()
        self.indexes = np.arange(len(self.training_data))
        self.batch_size = batch_size
        self.image_size = target_size
        self.shuffle = shuffle

    # 训练之前调用获取迭代次数
    def __len__(self):
        # 计算每一个epoch的迭代次数
        num_imgs = len(self.training_data)
        return math.ceil(num_imgs / float(self.batch_size))

    def __getitem__(self, idx):
        # 生成batch_size个索引
        batch_indexs = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        # 根据索引获取datas集合中的数据
        batch = [self.training_data[k] for k in batch_indexs]
        # 生成数据
        X, y = self.data_generation(batch, image_size=self.image_size, one_hot=True)
        return X, y
    
    def load_data(self):
        with open('CIFAR-10-train-label.pkl','rb') as f:
            training_data = pickle.load(f)
        return training_data
    
    def on_epoch_end(self):
        # 在每一次epoch结束是否需要进行一次随机，重新随机一下index
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def get_one_hot_label(self,labels,depth):
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

    def data_generation(self, value,image_size='NONE',depth=10,one_hot = False):
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
            y_batch = self.get_one_hot_label(y_batch,depth)
        return  np.array(x_batch,dtype=np.float32), np.array(y_batch,dtype=np.float32)


