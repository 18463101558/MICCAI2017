import tensorflow as tf
import tensorflow.contrib.slim as slim
from tflearn.layers.conv import global_avg_pool
#######################
# 3d functions
#######################
# convolution
def conv3d(input, output_chn, kernel_size, stride, use_bias=False, name='conv'):
    return tf.layers.conv3d(inputs=input, filters=output_chn, kernel_size=kernel_size, strides=stride,
                            padding='same', data_format='channels_last',
                            kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                            kernel_regularizer=slim.l2_regularizer(0.0005), use_bias=use_bias, name=name)
def fractal_conv3d(input, output_chn, kernel_size, stride, use_bias=False):#对于fractal来说，不需要重用权重
    return tf.layers.conv3d(inputs=input, filters=output_chn, kernel_size=kernel_size, strides=stride,
                            padding='same', data_format='channels_last',
                            kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                            kernel_regularizer=slim.l2_regularizer(0.0005), use_bias=use_bias)

def bn_relu_conv(input, output_chn, kernel_size, stride, use_bias, is_training, name):
    with tf.variable_scope(name):
        bn = tf.contrib.layers.batch_norm(input, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=is_training, scope="batch_norm")
        relu = tf.nn.relu(bn, name='relu')
        conv = conv3d(relu , output_chn, kernel_size, stride, use_bias, name='conv')
    return conv

def bn_relu(input,  is_training, name):
    with tf.variable_scope(name):
        bn = tf.contrib.layers.batch_norm(input, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=is_training, scope="batch_norm")
        relu = tf.nn.relu(bn, name='relu')
    return relu

# deconvolution
def Deconv3d(input, output_chn, name):
    batch, in_depth, in_height, in_width, in_channels = [int(d) for d in input.get_shape()]
    filter = tf.get_variable(name+"/filter", shape=[4, 4, 4, output_chn, in_channels], dtype=tf.float32,
                             initializer=tf.random_normal_initializer(0, 0.01), regularizer=slim.l2_regularizer(0.0005))

    conv = tf.nn.conv3d_transpose(value=input, filter=filter, output_shape=[batch, in_depth * 2, in_height * 2, in_width * 2, output_chn],
                                  strides=[1, 2, 2, 2, 1], padding="SAME", name=name)
    return conv
"""
def Unsample(input, output_chn, name):
    batch, in_depth, in_height, in_width, in_channels = [int(d) for d in input.get_shape()]
    base=input.shape[-2]
    data=96/int(base)
    print("base shape", data)
    filter = tf.get_variable(name+"/filter", shape=[4, 4, 4, output_chn, in_channels], dtype=tf.float32,
                             initializer=tf.random_normal_initializer(0, 0.01), regularizer=slim.l2_regularizer(0.0005))

    conv = tf.nn.conv3d_transpose(value=input, filter=filter, output_shape=[batch, 96, 96, 96, output_chn],
                                  strides=[1,data,data,data, 1], padding="SAME", name=name)
    return conv
"""
def  bn_relu_deconv(input, output_chn, is_training, name):
    with tf.variable_scope(name):
        bn = tf.contrib.layers.batch_norm(input, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=is_training, scope="batch_norm")
        relu = tf.nn.relu(bn, name='relu')
        conv = Deconv3d(relu, output_chn, name='deconv')
        # with tf.device("/cpu:0"):
    return conv
"""
def conv_bn_relu_x3(input, output_chn, kernel_size, stride, use_bias, is_training, name):
    with tf.variable_scope(name):
        z=conv_bn_relu(input, output_chn, kernel_size, stride, use_bias, is_training, "dense1")
        z_out = conv_bn_relu(z, output_chn, kernel_size, stride, use_bias, is_training, "dense2")
        z_out = conv_bn_relu(z_out, output_chn, kernel_size, stride, use_bias, is_training, "dense3")
        return z+z_out
"""
#通道数量不会变，就是一个1x1x1的卷积而已
def gate_block(input,output_chn, name):
    gate=conv_bn_relu(input, output_chn, kernel_size=1, stride=1, use_bias=False, is_training=1,name=name)
    #和图中的1x1x1卷积相同，用于执行一次粗略变换
    return gate

#返回多头注意力结果，第三个是门控通道数量，第四个是输入通道数量
#第一个是输入，第二个是门控，这一个就是综合两个的注意力
def MultiAttentionBlock(input, gate_signal, output_chn,name):
    gate_1 = GridAttentionBlock3D(input,gate_signal,inter_channels=output_chn,name=name+"att1")#inter_channels指输出的通道大小
    gate_2= GridAttentionBlock3D(input,gate_signal,inter_channels=output_chn,name=name+"att2")
    concat_1 = tf.concat([gate_1, gate_2], axis=4)
    avg_gate=(gate_1+gate_2)/2
    combine_gates =avg_gate + conv_bn_relu(concat_1,output_chn, kernel_size=1, stride=1, use_bias=False, is_training=1, name=name+"combine_gates")
    return combine_gates#再度引入残差连接

def GridAttentionBlock3D(input,gate_signal, inter_channels,name):
    theta_x = theta(input,inter_channels)#将输入尺寸缩小到原来1/2
    gate_signal=tf.stop_gradient(gate_signal)#阻止梯度更新到压缩后的gate处，
    # 也就是注意力机制只影响被更新的跳跃连接，不影响用于监控的gate_signal

    phi_g=phi(gate_signal,inter_channels)#将门控信号转换到和输入信号相同通道数量,由于这里gate本来就是原特征图一半大小，所以这里不用上采样
    add=theta_x+phi_g#利用局部特征图和全局特征图生成门控信号
    relu_gate = tf.nn.relu(add)
    sigmoid_gate=tf.sigmoid(psi(relu_gate))#转换成单通道特征图进行门控
    sigmoid_gate= Deconv3d(sigmoid_gate, inter_channels, name=name)#门控上采样到和输入相同大小
    y=sigmoid_gate*input#生成了注意力特征图,这里是和输入一样大小的
    w_y=W_Y(y,inter_channels)+y
    return w_y

#对输入信号进行变换
def theta(input,output_chn):
    #print("变换前shape：",input.shape) #1,12,12,12,512
    return tf.layers.conv3d(inputs=input, filters=output_chn, kernel_size=2, strides=2, data_format='channels_last',use_bias=False)
#对门控信号进行卷积
def phi(gate_signal,output_chn):
    return tf.layers.conv3d(inputs=gate_signal, filters=output_chn, kernel_size=1, strides=1,  data_format='channels_last',
                            use_bias=True)

def psi(input):
    return tf.layers.conv3d(inputs=input, filters=1, kernel_size=1, strides=1,  data_format='channels_last',
                           use_bias=True)
def W_Y(input,output_chn):
    conv1= tf.layers.conv3d(inputs=input, filters=output_chn, kernel_size=1, strides=1,
                            data_format='channels_last',use_bias=False)
    bn = tf.contrib.layers.batch_norm(conv1, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True,
                                      is_training=1)
    return bn

def Squeeze_Excitation_Block(input_x, output_chn, ratio=16):
    batch, in_depth, in_height, in_width, in_channels = [int( d ) for d in input_x.get_shape()]  # 取出各维度大小
    Squeeze= tf.reshape( input_x, (batch, in_depth , in_height * in_width,in_channels) )
    Squeeze = global_avg_pool(Squeeze)
    #print("squeeze",Squeeze.shape)#1+通道数量
    Excitation = tf.layers.dense(Squeeze, units=output_chn / ratio,use_bias=False)#进行压缩，学习注意参数
    Excitation = tf.nn.relu(Excitation)
    Excitation = tf.layers.dense( Excitation, units=output_chn,use_bias=False)
    Excitation = tf.sigmoid(Excitation)
    Excitation = tf.reshape(Excitation, [-1, 1, 1, 1,output_chn])#生成了缩放尺寸
    # print("ex:",Excitation.shape)
    # print("input_x:", input_x.shape)
    scale = input_x * Excitation#对输入进行缩放，美滋滋
    return scale