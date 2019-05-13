import tensorflow as tf


def bias_var(out_channels, init_method):
    initial_value = tf.constant(0.0, shape=[out_channels])
    biases = tf.Variable(initial_value)

    return biases


def conv_var(kernel_size, in_channels, out_channels, init_method, name):
    shape = [kernel_size[0], kernel_size[1], kernel_size[2], in_channels, out_channels]
    if init_method == 'msra':  # get_variable()，来说，如果已经创建的变量对象，就把那个对象返回，如果没有创建变量对象的话，就创建一个新的
        return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.variance_scaling_initializer())
    elif init_method == 'xavier':
        return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())


##if_b:if use bottleneck architecture
##channels_per_layer：每层滤波器数量
##layer_num=T/3:每一个块中总的层数
# keep_prob:不知道
# block_name：不知道
def clique_block(input_layer, channels_per_layer,outchannel, layer_num, is_train, keep_prob, block_name, loop_num=1):
    channels = channels_per_layer  # 每层滤波器数量
    #node_0_channels = input_layer.get_shape().as_list()[-1]  # 输入通道的channel大小shushi
    # print("node_0_channels:",node_0_channels) 等于64
    param_dict = {}
    kernel_size = (3, 3, 3)  # 使用1x1卷积或者3x3卷积
    for layer_id in range(1, layer_num):  # 对于每一层 产生该层到其他所有层对应的权重
        add_id = 1
        while layer_id + add_id <= layer_num:
            ## ->
            filters = conv_var(kernel_size=kernel_size, in_channels=channels, out_channels=channels, init_method='msra',
                               name=block_name + '-' + str(layer_id) + '_' + str(layer_id + add_id))
            # 根据name找到对应的滤波器核 块-层-layer_id+add_id来构成应该是哪一组滤波器
            param_dict[str(layer_id) + '_' + str(layer_id + add_id)] = filters
            # 找到这个滤波器核对应的位置

            ## <-
            filters_inv = conv_var(kernel_size=kernel_size, in_channels=channels, out_channels=channels,
                                   init_method='msra',
                                   name=block_name + '-' + str(layer_id + add_id) + '_' + str(layer_id))
            param_dict[str(layer_id + add_id) + '_' + str(layer_id)] = filters_inv
            # 根据name找到对应的滤波器核 add_id+块-layer_id+add_id-层来构成应该是哪一组滤波器，这两货找到的东西还不太一样
            add_id += 1  # 10*2=20#我寻思这个应该指哪两个层应该连接的权重

    for layer_id in range(layer_num):  # 从输入层通往其他所有层
        filters = conv_var(kernel_size=kernel_size, in_channels=channels, out_channels=channels,
                           init_method='msra', name=block_name + '-' + str(0) + '_' + str(layer_id + 1))
        param_dict[str(0) + '_' + str(layer_id + 1)] = filters  # 生成第三组id，但是在这里怎么用尚且不知道
        # print(filters.shape)
    # print("param_dict:",list(sorted(param_dict.keys())))
    assert len(param_dict) == layer_num * (layer_num - 1) + layer_num  # 这玩意等于块中总层数的平方，麻蛋带进去算还真是

    ## init blob
    blob_dict = {}

    for layer_id in range(1, layer_num + 1):  # 1-5，但是第一层不会进去
        bottom_blob = input_layer
        bottom_param = param_dict['0_' + str(layer_id)]  # 取出参数列表，噗哈哈哈
        for layer_id_id in range(1, layer_id):  # 第一次不进入循环，但是将bottom_blob弄为等于input_layer，
            # 第二次循环呢，就把大小变成了多加一个快，所以第二次输出就是等于densenet第二个块了，然后依次第三个，第四个

            # print("layer_id_id", layer_id_id, ",bottom_param:", bottom_param.shape)
            bottom_blob = tf.concat((bottom_blob, blob_dict[str(layer_id_id)]), axis=4)  # 顺序依次生成densenet块
            bottom_param = tf.concat((bottom_param, param_dict[str(layer_id_id) + '_' + str(layer_id)]),
                                     axis=3)  # 这个原来是滤波器
            # 这里分别用的是0_1;;0_2 1_2;;0_3 1_3 2_3;;0_4 1_4 2_4 3_4;;0_5 1_5 2_5 3_5 4_5这几个滤波器

        # bottom_blob这里输出的结果就是densenet的块
        mid_layer = tf.contrib.layers.batch_norm(bottom_blob, scale=True, is_training=is_train,
                                                 updates_collections=None)
        mid_layer = tf.nn.relu(mid_layer)
        mid_layer = tf.nn.conv3d(mid_layer, bottom_param, [1, 1, 1, 1, 1], padding='SAME')
        # print(bottom_param.shape)[filter_height, filter_width, in_channels, out_channels]
        mid_layer = tf.nn.dropout(mid_layer, keep_prob)  # shape始终等于64
        # print("midlayer:",mid_layer.shape)#
        next_layer = mid_layer
        blob_dict[str(layer_id)] = next_layer  # dense net的输出结果

    ## begin loop 也就是构建循环阶段
    for loop_id in range(loop_num):
        for layer_id in range(1, layer_num + 1):  ##   [1,2,3,4,5]

            layer_list = [str(l_id) for l_id in range(1, layer_num + 1)]  ##   [1,2,3,4,5]
            layer_list.remove(str(layer_id))  # 删除掉对应当前的

            bottom_blobs = blob_dict[layer_list[0]]
            bottom_param = param_dict[layer_list[0] + '_' + str(layer_id)]

            # print("bottom_blobs:",layer_list[0] ) 第一步是更新第一个块的对象，这时候需要所有2,3,4,5来为它产生更新用的值
            # print("bottom_param:",layer_list[0]+'_'+str(layer_id))
            for bottom_id in range(len(layer_list) - 1):
                bottom_blobs = tf.concat((bottom_blobs, blob_dict[layer_list[bottom_id + 1]]),
                                         axis=4)  ###  concatenate the data blobs
                # print("bottom_blobs:", layer_list[bottom_id+1])
                bottom_param = tf.concat((bottom_param, param_dict[layer_list[bottom_id + 1] + '_' + str(layer_id)]),
                                         axis=3)  ###  concatenate the parameters       注意这里改变的也是inchannel哦
                # print("bottom_param:", layer_list[bottom_id+1]+'_'+str(layer_id))
            #print(bottom_param.shape)  # (3, 3, 256, 64)
            # print("end")
            mid_layer = tf.contrib.layers.batch_norm(bottom_blobs, scale=True, is_training=is_train,
                                                     updates_collections=None)
            mid_layer = tf.nn.relu(mid_layer)
            mid_layer = tf.nn.conv3d(mid_layer, bottom_param, [1, 1, 1, 1, 1],
                                     padding='SAME')  ###  update the data blob
            mid_layer = tf.nn.dropout(mid_layer, keep_prob)

            next_layer = mid_layer
            blob_dict[str(layer_id)] = next_layer

    transit_feature = blob_dict['1']  # 这是第一层输出的结果吧

    for layer_id in range(2, layer_num + 1):
        transit_feature = tf.concat((transit_feature, blob_dict[str(layer_id)]), axis=4)

    block_feature = tf.concat((input_layer, transit_feature), axis=4)#densenet块产生的权重

    block_feature_layer = tf.contrib.layers.batch_norm(block_feature , scale=True, is_training=is_train,
                                             updates_collections=None)
    block_feature_layer = tf.nn.relu(block_feature_layer)
    filters = conv_var(kernel_size=(1,1,1), in_channels=block_feature_layer.get_shape().as_list()[-1] , out_channels=outchannel, init_method='msra',
                       name=block_name + '-' +"changechannel")
    block_feature =tf.nn.conv3d(block_feature_layer ,filters, [1, 1, 1, 1, 1], padding='SAME')
    #print(block_feature.shape)
    # print( transit_feature.shape)#这特么就变成320了
    return block_feature
