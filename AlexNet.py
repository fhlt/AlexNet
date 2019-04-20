import tensorflow as tf

def conv_op(name, inputs, kernel_h, kernel_w, out_channels, strides, padding="SAME", is_activation = True):
    with tf.name_scope(name) as scope:
        in_channels = inputs.get_shape()[-1].value
        kernel = tf.get_variable(name=scope, shape=[kernel_h, kernel_w, in_channels, out_channels],
                                    dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(inputs, kernel, strides=(1, strides, strides, 1), padding=padding)
        b_init = tf.constant(0, shape=[out_channels], dtype=tf.float32)
        biases = tf.Variable(b_init, trainable=True)
        z = tf.nn.bias_add(conv, biases)
        # 在此处添加batch_normalization
        # z = tf.layers.batch_normalization(z)
        #z = tf.nn.batch_normalization(z)
        if is_activation:
            activation = tf.nn.relu(z)
        else:
            activation = z
        '''
        if is_pooling:
            output = tf.nn.max_pool(value=activation, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        else:
            output = activation
        '''
    return activation
def max_pool(name, inputs, filter_height, filter_width, stride_y, stride_x,padding="SAME"):
    return tf.nn.max_pool(value=inputs,
                          ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding="SAME", name=name)

def local_response_normalized(x, radius, alpha, beta, name, bias=1.0):
    return tf.nn.local_response_normalization(x, depth_radius=radius,
                                              alpha=alpha, beta=beta,
                                              bias=bias, name=name)

# inputs is a placeholder (with shape of [?, 224, 224, 3])
def AlexNet(inputs, n_classes, keep_prob):
    '''
    Create the network graph.
    创建AlexNet网络 返回网络输出的张量(没有经过正激函数)
    共有8层网络  5层卷积测，3个全连接网络
    '''
    # 1st Layer: Conv (w ReLu) -> Lrn -> Pool  --------> (55, 55, 96)
    conv1 = conv_op("conv1", inputs, 11, 11, 96, 4)
    norm1 = local_response_normalized(conv1, radius=4, alpha=1e-4, beta=0.75,  name='norm1')
    # (27,27,96)
    pool1 = max_pool("pool1", norm1, filter_height=3, filter_width=3, stride_y=2, stride_x=2, padding='VALID')
    # 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool with 2 groups ------> (27, 27, 256)
    conv2 = conv_op("conv2", pool1, 5, 5, 256, 1)
    norm2 = local_response_normalized(conv2, radius=4, alpha=1e-4, beta=0.75, name='norm2')
    # (13, 13, 256)
    pool2 = max_pool("pool2", norm2, 3, 3, 2, 2, padding='VALID')
    # 3rd Layer: Conv (w ReLu)  -------> (13, 13, 384)
    conv3 = conv_op("conv3", pool2, 3, 3, 384, 1)
    # 4th Layer: Conv (w ReLu) splitted into two groups   ---->  (13, 13, 384)
    conv4 = conv_op("conv4", conv3, 3, 3, 384, 1)   
    # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups   ---->  (13, 13, 256)
    conv5 = conv_op("conv5", conv4, 3, 3, 256, 1)
    # (6, 6, 256)
    pool5 = max_pool("pool5", conv5, 3, 3, 2, 2, padding="VALID")

    flatten = tf.layers.flatten(pool5)   # (?, 6*6*256)
    # 第一层全连接
    fc1 = tf.layers.dense(inputs=flatten, units=4096,activation=tf.nn.relu, name="FC1")
    drop1 = tf.layers.dropout(fc1, keep_prob)
    # 第二层全连接
    # 修改AlexNet（4096------>2048）
    fc2 = tf.layers.dense(inputs=drop1, units=2048,activation=tf.nn.relu, name="FC2")
    drop2 = tf.layers.dropout(fc2, keep_prob)
    # 第二层全连接
    # 最后一层不激活直接输出
    fc3 = tf.layers.dense(inputs=drop2, units=n_classes,activation=None, name="FC3")
    return fc3

# 计算交叉熵损失
def AlexNet_loss(y,y_hat):
    # 损失
    y_hat = tf.cast(y_hat, dtype=tf.float32)
    # 先进行softmax，然后计算交叉熵
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y,labels = y_hat))
    # 准确率
    # softmax不影响相对大小
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1),tf.argmax(y_hat,1)), tf.float32))
    return loss, accuracy
