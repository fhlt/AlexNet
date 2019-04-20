import tensorflow as tf
from AlexNet import AlexNet_loss, AlexNet
from tqdm import tqdm
import os
from generator import BatchGenerator

# 训练集采用generator的方式产生
# 测试集采用yield的方式产生

LEARNINF_RATE = 1e-4                                 #学习率
EPOCHES = 10                                         #训练轮数
BATCH_SIZE = 128                                     #批量大小
N_CLASSES = 10                                       #标签的维度
DROPOUT = 0.5                                        #dropout概率
IMAGE_H = 224                                        #图片大小
IMAGE_W = 224

model_path = "model"
model_name = "cifar10.ckpt"
# 训练集数据生成器（测试集采用yield的方式产生）
train_data = BatchGenerator(target_size=(IMAGE_H, IMAGE_W), batch_size=BATCH_SIZE)

# 创建placeholder
inputs_placeholder = tf.placeholder(dtype=tf.float32,shape=[None, IMAGE_H, IMAGE_W, 3], name="INPUTS")
# 理论上这里写tf.int32更合适,写tf.float32也行
labels_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, N_CLASSES], name="INPUTS")
keep_prob_placeholder = tf.placeholder(tf.float32,name='keep_prob')

alex_outputs = AlexNet(inputs_placeholder, N_CLASSES, keep_prob_placeholder)
cost, accuracy = AlexNet_loss(alex_outputs, labels_placeholder)

train_op = tf.train.AdamOptimizer(LEARNINF_RATE).minimize(cost)
#创建Saver op用于保存和恢复所有变量
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())     #初始化变量
    #如果该文件存在，恢复模型数据
    #if os.path.isfile(+'.meta'):
    #    saver.restore(sess,model_name)
    for epoch in range(EPOCHES):
        print("epoche:",epoch)
        for i in tqdm(range(len(train_data))):
            X, y = train_data.__getitem__(i)
            _, loss, acc = sess.run([train_op, cost, accuracy], feed_dict={inputs_placeholder:X,
                                          labels_placeholder:y,
                                          keep_prob_placeholder:DROPOUT})
            print("loss:%f,acc:%f" % (loss, acc))
        train_data.on_epoch_end()
    # 全部训练结束，保存模型
    saver.save(sess, os.path.join(model_path, model_name))
            
            
