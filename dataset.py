import pickle
import numpy as np
import cv2
from skimage import io
import os
'''
名称              作用
b'data’          是一个10000x3072的array，每一行的元素组成了一个32x32的3通道图片，共10000张
b'labels’          一个长度为10000的list，对应包含data中每一张图片的label
b'batch_label' 这一份batch的名称
b'filenames'      一个长度为10000的list，对应包含data中每一张图片的名称
'''
data_dir = "data"
data_dir_cifar100 = os.path.join(data_dir, "cifar-100-python")
data_dir_cifar10 = os.path.join(data_dir, "cifar-10-python")

class_names_cifar100 = np.load(os.path.join(data_dir_cifar100, "meta"))
class datagenerator(object):
    def __init__(self):
        pass
        
        
    def unpickle(self,filename):
        '''
        batch文件中真正重要的两个关键字是data和labels        
        反序列化出对象
        
        每一个batch文件包含一个python的字典（dict）结构，结构如下：
        名称              作用
        b'data’          是一个10000x3072的array，每一行的元素组成了一个32x32的3通道图片，共10000张
        b'labels’          一个长度为10000的list，对应包含data中每一张图片的label
        b'batch_label' 这一份batch的名称
        b'filenames'      一个长度为10000的list，对应包含data中每一张图片的名称        
        '''
        with open(filename,'rb')  as  f:
            #默认把字节转换为ASCII编码  这里设置encoding='bytes'直接读取字节数据  因为里面含有图片像素数据 大小从0-255 不能解码为ascii编码,因此先转换成字节类型 后面针对不同项数据再解码，转换为字符串        
            dic = pickle.load(f,encoding='bytes')     
        return dic
    
    
    def get_image(self,image):
        '''
        提取每个通道的数据，进行重新排列，最后返回一张32x32的3通道的图片
        
        在字典结构中，每一张图片是以被展开的形式存储（即一张32x32的3通道图片被展开成了3072长度的list），
        每一个数据的格式为uint8，前1024个数据表示红色通道，接下来的1024个数据表示绿色通道，最后的1024
        个通道表示蓝色通道。
        image:每一张图片的数据  数据按照R,G,B通道依次排列 长度为3072
        '''
        assert len(image) == 3072
        #对list进行切片操作,然后reshape
        r = image[:1024].reshape(32,32,1)
        g = image[1024:2048].reshape(32,32,1)
        b = image[2048:].reshape(32,32,1)
        
        #numpy提供了numpy.concatenate((a1,a2,...), axis=0)函数。能够一次完成多个数组的拼接。其中a1,a2,...是数组类型的参数
        #沿着某个轴拼接，默认为列方向（axis=0）
        img = np.concatenate((r,g,b),-1)
        return img
        
    
    def get_data_by_keyword(self,keyword,filelist=[],normalized=False,size=(32,32),one_hot=False):
        '''
        按照给出的关键字提取batch中的数据（默认是训练集的所有数据）
        
        args:
            keyword：'data’ 或 'labels’ 或  'batch_label' 或  'filenames' 表示需要返回的项
            filelist：list 表示要读取的文件集合
            normalized：当keyword = 'data'，表示是否需要归一化
            size：当keyword = 'data'，表示需要返回的图片的尺寸
            one_hot:当keyword = 'labels'时，one_hot=Flase，返回实际标签  True时返回二值化后的标签
            
        return:
            keyword = 'data' 返回像素数据
            keyword = 'labels' 返回标签数据
            keyword = 'batch_label' 返回batch的名称
            keyword = 'filenames' 返回图像文件名
                 
        '''
        
        #keyword编码为字节
        keyword = keyword.encode('ascii')
        assert keyword in [b'data',b'labels',b'batch_label',b'filenames']
        assert type(filelist) is list and len(filelist) != 0
        assert type(normalized) is bool
        assert type(size) is tuple or type(size) is list
                
        
        ret = []
        
             
        for i in range(len(filelist)):            
            #反序列化出对象
            dic = self.unpickle(filelist[i])
            
            if keyword == b'data':
                #b'data’          是一个10000x3072的array，每一行的元素组成了一个32x32的3通道图片，共10000张
                #合并成一个数组           
                for item in dic[b'data']:
                    ret.append(item) 
                print('总长度:',len(ret))
                
            elif keyword == b'labels':
                #b'labels’          一个长度为10000的list，对应包含data中每一张图片的label          
                #合并成一个数组               
                for item in dic[b'labels']:
                    ret.append(item) 
              
                
            elif keyword == b'batch_label':
                #b'batch_label' 这一份batch的名称           
                #合并成一个数组               
                for item in dic[b'batch_label']:
                    ret.append(item.decode('ascii'))    #把数据转换为ascii编码
            
            else:
                #b'filenames'      一个长度为10000的list，对应包含data中每一张图片的名称                    
                #合并成一个数组               
                for item in dic[b'filenames']:
                    ret.append(item.decode('ascii'))    #把数据转换为ascii编码
                
            
        if keyword == b'data':                      
            if normalized == False:
                array = np.ndarray([len(ret),size[0],size[1],3],dtype = np.float32)                
                #遍历每一张图片数据
                for i in range(len(ret)):
                    #图像进行缩放
                    array[i] = cv2.resize(self.get_image(ret[i]),size)
                return array
            
            else:
                array = np.ndarray([len(ret),size[0],size[1],3],dtype = np.float32)                
                #遍历每一张图片数据
                for i in range(len(ret)):
                    array[i] = cv2.resize(self.get_image(ret[i]),size)/255
                return array
            pass
        
        elif keyword == b'labels':
            #二值化标签
            if one_hot == True:
                #类别
                depth = 10
                m = np.zeros([len(ret),depth])
                for i in range(len(ret)):
                    m[i][ret[i]] = 1
                return m
            pass
        #其它keyword直接返回
        return ret


def  save_images():
    '''
    报CIFAR-10数据集图片提取出来保存下来  
    1.创建一个文件夹 CIFAR-10-data 包含两个子文件夹test,train
    2.在该文件夹创建10个文件夹 文件名依次为0-9  对应10个类别
    3.训练集数据生成bmp格式文件，存在对应类别的文件下    
    4.测试集数据生成bmp格式文件，存在对应类别的文件下  
    5 生成两个文件train_label.pkl，test_label.pkl 分别保存相应的图片文件路径以及对应的标签
    '''
    
    #根目录
    root = 'CIFAR-10-data'
    
    #如果存在该目录 说明数据存在
    if os.path.isdir(root):
        print(root+'目录已经存在!')
    else:
        #'data'目录不存在，创建目录
        os.mkdir(root)
    
    #创建文件失败
    if not os.path.isdir(root): 
        print(root+'目录创建失败!')
        return 
    
    #创建'test'和'train'目录  以及子文件夹
    train = os.path.join(root,'train')
    os.mkdir(train)
    if os.path.isdir(train):
        for i in range(10):
            name = os.path.join(train,str(i))
            os.mkdir(name)
            
    test = os.path.join(root,'test')
    os.mkdir(test)
    if os.path.isdir(test):
        for i in range(10):
            name = os.path.join(test,str(i))
            os.mkdir(name)
            
    
    
    '''
    把训练集数据转换为图片
    '''
    data_dir = data_dir_cifar10       #数据所在目录
    
    filelist = []
    for i in range(5):
        name = os.path.join(data_dir,str('data_batch_%d'%(i+1)))
        filelist.append(name)
        
    data = datagenerator()
    #获取训练集数据
    train_x = data.get_data_by_keyword('data',filelist,
                                           normalized=True,size=(32,32)) 
        
    #标签
    train_y = data.get_data_by_keyword('labels',filelist)
                                           
    
    #读取图片文件名
    train_filename  = data.get_data_by_keyword('filenames',filelist)
                                            
    
   
    #保存训练集的文件名和标签
    train_file_labels  = [] 
    
    #保存图片
    for i in  range(len(train_x)):
        #获取图片标签
        y = int(train_y[i])
        
        #文件保存目录
        dir_name = os.path.join(train,str(y))
  
        #获取文件名
        file_name = train_filename[i]       
       
        #文件的保存路径
        file_path = os.path.join(dir_name,file_name)
     
        #保存图片
        io.imsave(file_path,train_x[i])
        
        #追加第i张图片路径和标签   (文件路径,标签)
        train_file_labels.append((file_path,y))
        
        if i % 1000 == 0:
            print('训练集完成度{0}/{1}'.format(i,len(train_x)))
    
    for i in range(10):
        print('训练集前10张图片：',train_file_labels[i])
        
    #保存训练集的文件名和标签
    with open('CIFAR-10-train-label.pkl','wb') as f:      
        pickle.dump(train_file_labels,f)
    
    print('训练集图片保存成功!\n')
    
    '''
    把测试集数据转换为图片
    '''
    filelist = [os.path.join(data_dir,'test_batch')]
    #获取训练集数据 数据标准化为0-1之间
    test_x = data.get_data_by_keyword('data',filelist,
                                           normalized=True,size=(32,32)) 
        
    #标签
    test_y = data.get_data_by_keyword('labels',filelist)                                      
    #读取图片文件名
    test_filename = data.get_data_by_keyword('filenames',filelist)
    #保存测试卷的文件名和标签    
    test_file_labels  = [] 
     
    #保存图片
    for i in  range(len(test_x)):
        #获取图片标签
        y = int(test_y[i])
        #文件保存目录
        dir_name = os.path.join(test,str(y))
        #获取文件名
        file_name = test_filename[i]     
        #文件的保存路径
        file_path = os.path.join(dir_name,file_name)
        #保存图片  这里要求图片像素值在-1-1之间，所以在获取数据的时候做了标准化
        io.imsave(file_path,test_x[i])
        #追加第i张图片路径和标签  (文件路径,标签)
        test_file_labels.append((file_path,y))
        if i % 1000 == 0:
            print('测试集完成度{0}/{1}'.format(i,len(test_x)))
    
    print('测绘集图片保存成功!\n')
    
    #保存测试卷的文件名和标签
    with open('CIFAR-10-test-label.pkl','wb') as f:        
        pickle.dump(test_file_labels,f)   
    
    for i in range(10):
        print('测试集前10张图片：',test_file_labels[i])
'''
测试 保存所有图片
'''
#save_images()









        
