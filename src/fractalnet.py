import numpy as np
import tensorflow as tf
from ops import *


def get_join_layer(row,mask,is_global_path,global_path_list,local_path_list,input):
    """
    """
    def global_path_merge():
        count = len(mask)#和柱数量相当
        ave=input[0]*global_path_list[0]
        for i in range(1, count):
            ave+=input[i]*global_path_list[i]
        return ave/tf.reduce_sum(global_path_list)#全局路径中始终只有一个列为1，如果该列在其中，那么合并后的值就是该列输入的值，如果不在，那么合并后的值为0

    def has_local_path_mask():#新的mask之后
        count = len(mask)  # 和柱数量相当
        new_mask = local_path_list * mask
        ave = input[0] * new_mask[0]
        for i in range(1, count):
            ave += input[i] * new_mask[i]
        return ave / tf.reduce_sum(new_mask)

    def no_local_path():
        ave = input[0] * mask[0]
        for i in range(1, len(mask)):
            ave += input[i] * mask[i]
        return ave / tf.reduce_sum(tf.to_float(mask))#返回的其实就是均值合并的结果

    def local_path_merge():#每次合并时，保证至少有一条路径
        new_mask=local_path_list*mask#she
        has_path=tf.reduce_sum(new_mask)
        ave=tf.cond( has_path>0.0 ,has_local_path_mask,no_local_path)#条件满足，就会进入前面那个筛选出的局部路径
        return ave

    def f():
        #输入的是一个多个待合并的，因为这多个块大小相同，所以[0],又因为batchshape不考虑，所以1： [32, 32, 64]
        product = tf.cond(is_global_path>0, global_path_merge,local_path_merge)#条件满足，就会产生前面那个，进入全局merge
        return product
    return f

def fractal_conv():
    def f(prev):
        conv =  fractal_conv3d(input=prev, output_chn=8, kernel_size=3, stride=1, use_bias=False)
        dropout = None
        if dropout:
            conv = tf.nn.dropout( conv, dropout)
        conv_bn = tf.contrib.layers.batch_norm(conv, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True,is_training=1)
        conv_relu =  tf.nn.relu(conv_bn)
        return conv_relu
    return f

def fractal_block(Columns,is_global_path,global_path_list,local_path_list):
    def f(z):
        columns = [[z] for _ in range(Columns)]#c这里等于3，所以这里将z复制了三份，成为一个新数组，代表一共有三条路径
        for row in range(2**(Columns-1)):#网络深度从0到2**(c-1) - 1 先遍历深度
            t_row = []
            for col in range(Columns):
                prop = 2**(col)#该深度对应的层是否有卷积的必要
                #对于当前块，在这一层是否有必要卷积
                if (row+1) % prop == 0:#因为index是从0开始的，所以得加1，并且比如对于col=2，也就是最后一个，那么他只有一次卷积的机会
                    t_col = columns[col]
                    t_col.append(fractal_conv()(t_col[-1]))  # 执行对应的卷积，注意columns外面本来就有一个中括号，所以这里相当于是columns[col]再串联了一下
                    t_row.append(col)#t_row代表合并最后一层

            merging=[]
            mask=[]
            if len(t_row) > 1:#注意不需要合并的在if这里就过滤掉了
                for i in range(Columns):
                    if i in t_row:
                        mask.append(1)
                        merging.append(columns[i][-1])#这一个-1有点意思，是因为这里 [[z] for _ in range(c)]，z外面有两个中括号，所以取出来的也就是最后面一层的结果
                    else:
                        mask.append(0)
                        merging.append(0)
                merged  = get_join_layer(row,mask,is_global_path,global_path_list,local_path_list[row],merging)()#刚刚得到的是两个块，现在打算将两个块合并起来 针对每个块的路径一样
                for i in t_row:
                    columns[i].append(merged)#把合并好的那个贴到最后面
        return columns[0][-1]#当然是选择返回最后一个值啦
    return f

def fractal_net(is_global_path_list,global_path_list,local_path_list,Blocks,Columns):
    def f(z):
        output = z
        for i in range(Blocks):#Blocks是堆叠块的个数-
            output=fractal_block(Columns=Columns,is_global_path=is_global_path_list[i],global_path_list=global_path_list[i],local_path_list=local_path_list[i])(output)#drop率
        return output
    return f
