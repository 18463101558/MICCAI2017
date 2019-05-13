from __future__ import division
import os
import time
from glob import glob
import cv2
import scipy.ndimage
from ops import *
from utils import *
from seg_eval import *
from fractalnet import fractal_net
from cliquenet import *
import tensorflow as tf
class unet_3D_xy(object):
    """ Implementation of 3D U-net"""
    def __init__(self, sess, param_set):
        self.sess           = sess
        self.phase          = param_set['phase']
        self.batch_size     = param_set['batch_size']
        self.inputI_size    = param_set['inputI_size']
        self.inputI_chn     = param_set['inputI_chn']
        self.outputI_size   = param_set['outputI_size']
        self.output_chn     = param_set['output_chn']
        self.resize_r       = param_set['resize_r']
        self.traindata_dir  = param_set['traindata_dir']
        self.chkpoint_dir   = param_set['chkpoint_dir']
        self.lr             = param_set['learning_rate']
        self.beta1          = param_set['beta1']
        self.epoch          = param_set['epoch']
        self.model_name     = param_set['model_name']
        self.save_intval    = param_set['save_intval']
        self.testdata_dir   = param_set['testdata_dir']
        self.labeling_dir   = param_set['labeling_dir']
        self.ovlp_ita       = param_set['ovlp_ita']
        self.step = param_set['step']
        self.rename_map = param_set['rename_map']
        self.rename_map = [int(s) for s in self.rename_map.split(',')]
        self.Blocks=param_set['Blocks']
        self.Columns=param_set['Columns']
        self.Stages=param_set['Stages']
        # build model graph
        self.build_model()#在这里开始建立网络

    # dice loss function
    def dice_loss_fun(self, pred, input_gt):
        input_gt = tf.one_hot(input_gt, 8)
        #############################################################
        #softmaxpred = tf.nn.softmax(pred)
        # input_gt = produce_mask_background(input_gt,softmaxpred )
        ####################################################################
        dice = 0
        for i in range(8):
            inse = tf.reduce_mean(pred[:, :, :, :, i]*input_gt[:, :, :, :, i])
            l = tf.reduce_sum(pred[:, :, :, :, i]*pred[:, :, :, :, i])
            r = tf.reduce_sum(input_gt[:, :, :, :, i] * input_gt[:, :, :, :, i])
            dice = dice + 2*inse/(l+r)
        return -dice

    # class-weighted cross-entropy loss function
    def softmax_weighted_loss(self, logits, labels):
        """
        Loss = weighted * -target*log(softmax(logits))
        :param logits: probability score
        :param labels: ground_truth
        :return: softmax-weifhted loss
        """
        gt = tf.one_hot(labels, 8)
        pred = logits
        softmaxpred = tf.nn.softmax(pred)
        #############################################################
        gt = produce_mask_background(gt, softmaxpred)#根据预测值生成对grountruth的掩膜
        gt = tf.stop_gradient( gt )  # 不需要对groundtruth产生梯度
        ####################################################################
        loss = 0
        for i in range(8):
            gti = gt[:,:,:,:,i]
            predi = softmaxpred[:,:,:,:,i]
            weighted = 1-(tf.reduce_sum(gti)/tf.reduce_sum(gt))
            #focal_loss=tf.pow( (1-tf.clip_by_value(predi, 0.005, 1)) , 4, name=None)
            focal_loss=1
            loss = loss + -tf.reduce_mean(weighted * gti *focal_loss* tf.log(tf.clip_by_value(predi, 0.005, 1)))
        return loss

    # build model graph
    def build_model(self):
        # input
        self.input_I = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.inputI_size, self.inputI_size, self.inputI_size, self.inputI_chn], name='inputI')
        print("输入层：",self.input_I)
        self.input_gt = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.inputI_size, self.inputI_size, self.inputI_size], name='target')
        print("输出层：", self.input_I)#输入输出都是1 96 96 1

        self.is_global_path= tf.placeholder(dtype=tf.float32, shape=[self.Stages, self.Blocks])
        self.global_path_list= tf.placeholder(dtype=tf.float32, shape=[self.Stages, self.Blocks, self.Columns])
        self.local_path_list= tf.placeholder(dtype=tf.float32, shape=[self.Stages, self.Blocks,2**(self.Columns-1),self.Columns ])

        self.pred_prob, self.pred_label, self.aux0_prob, self.aux1_prob, self.aux2_prob = self.unet_3D_model(self.input_I,self.is_global_path,self.global_path_list,self.local_path_list)#网络主体结构
        # ========= dice loss
        self.main_dice_loss = self.dice_loss_fun(self.pred_prob, self.input_gt)
        # auxiliary loss
        self.aux0_dice_loss = self.dice_loss_fun(self.aux0_prob, self.input_gt)
        self.aux1_dice_loss = self.dice_loss_fun(self.aux1_prob, self.input_gt)
        self.aux2_dice_loss = self.dice_loss_fun(self.aux2_prob, self.input_gt)
        self.total_dice_loss = self.main_dice_loss + 0.2*self.aux0_dice_loss + 0.4*self.aux1_dice_loss + 0.8*self.aux2_dice_loss

        # ========= class-weighted cross-entropy loss
        self.main_wght_loss = self.softmax_weighted_loss(self.pred_prob, self.input_gt)
        self.aux0_wght_loss = self.softmax_weighted_loss(self.aux0_prob, self.input_gt)
        self.aux1_wght_loss = self.softmax_weighted_loss(self.aux1_prob, self.input_gt)
        self.aux2_wght_loss = self.softmax_weighted_loss(self.aux2_prob, self.input_gt)
        self.total_wght_loss = self.main_wght_loss + 0.3*self.aux0_wght_loss + 0.6*self.aux1_wght_loss + 0.9*self.aux2_wght_loss

        self.total_loss = self.total_wght_loss
        # self.total_loss = self.total_wght_loss

        # trainable variables 返回的是需要训练的变量列表
        self.u_vars = tf.trainable_variables()

        # extract the layers for fine tuning
        ft_layer = ['conv1/kernel:0',
                    'conv2/kernel:0',
                    'conv3a/kernel:0',
                    'conv3b/kernel:0',
                    'conv4a/kernel:0',
                    'conv4b/kernel:0']

        self.ft_vars = []
        for var in self.u_vars:
            for k in range(len(ft_layer)):
                if ft_layer[k] in var.name:
                    self.ft_vars.append(var)#把这玩意作为变量名称放进去
                    break

        # create model saver
        self.saver = tf.train.Saver()
        # saver to load pre-trained C3D model 设置新的saver
        self.saver_ft = tf.train.Saver(self.ft_vars)

    # 3D unet graph
    def unet_3D_model(self, inputI,is_global_path_list,global_path_list,local_path_list):
        """3D U-net"""
        phase_flag = 1
        concat_dim = 4
        layer_num=3
        keep_prob=0.8#控制drop率

        conv1_1 = conv3d(input=inputI, output_chn=64, kernel_size=3, stride=1, use_bias=False, name='conv1')
        print("conv1_1:", conv1_1.shape)# pool1 (1, 96, 96, 96, 64)

        pool1= tf.layers.max_pooling3d(inputs=conv1_1, pool_size=2, strides=2, name='pool1')
        print("pool1:", pool1.shape)# pool1 (1, 48, 48, 48, 64)
        conv2_1 = clique_block(pool1, channels_per_layer=64, outchannel=128,layer_num=layer_num, is_train=phase_flag,
                                                    keep_prob=keep_prob, block_name='b' + str(1))#注意输入的是卷积后的结果，输出也是卷积后的结果
        print("conv2_1:",conv2_1.shape)#(1, 48, 48, 48, 128)

        pool2 = tf.layers.max_pooling3d(inputs=conv2_1, pool_size=2, strides=2, name='pool2')
        print("pool2:", pool2.shape)#pool2  (1, 24, 24, 24, 128)
        conv3_1 = clique_block(pool2, channels_per_layer=128, outchannel=256,layer_num=layer_num, is_train=phase_flag,
                                                    keep_prob=keep_prob, block_name='b' + str(2))#注意输入的是卷积后的结果，输出也是卷积后的结果
        print("conv3_1:", conv3_1.shape)#(1, 24, 24, 24, 256)


        pool3 = tf.layers.max_pooling3d(inputs=conv3_1, pool_size=2, strides=2, name='pool3')
        print("pool3:", pool3.shape)# (1, 12, 12,12, 256)
        conv4_1 = clique_block(pool3, channels_per_layer=256,outchannel=512, layer_num=layer_num, is_train=phase_flag,
                               keep_prob=keep_prob, block_name='b' + str(3))  # 注意输入的是卷积后的结果，输出也是卷积后的结果
        print("conv4_1:", conv4_1.shape)# (1, 12, 12,12, 512)

        pool4 = tf.layers.max_pooling3d(inputs=conv4_1, pool_size=2, strides=2, name='pool4')
        print("pool4:", pool4.shape)#pool4 (1, 6, 6, 6, 512)
        conv5_1 = clique_block(pool4, channels_per_layer=512,outchannel=512, layer_num=layer_num, is_train=phase_flag,
                               keep_prob=keep_prob, block_name='b' + str(4))  # 注意输入的是卷积后的结果，输出也是卷积后的结果
        print("conv5_1:", conv5_1.shape)#conv5_1(1, 6, 6, 6, 512)

        deconv1_1 = deconv_bn_relu(input=conv5_1, output_chn=512, is_training=phase_flag, name='deconv1_1')#注意BN在前面，CONV在最后面
        print("deconv1_1:", deconv1_1.shape)#(1, 12, 12, 12, 512)
        concat_1 = tf.concat([deconv1_1, conv4_1], axis=concat_dim, name='concat_1')
        print("concat_1:", concat_1.shape)#(1, 12, 12, 12, 1024)

        concat_1=conv3d(concat_1, 256, kernel_size=1, stride=1, use_bias=False, name='concat_1_short')
        deconv1_2 = clique_block(concat_1, channels_per_layer=256,outchannel=256, layer_num=layer_num, is_train=phase_flag,
                               keep_prob=keep_prob, block_name='b' + str(5))  # 注意输入的是卷积后的结果，输出也是卷积后的结果
        print("deconv1_2:", deconv1_2.shape)# (1, 12, 12, 12, 256)

        deconv2_1 = deconv_bn_relu(input=deconv1_2, output_chn=256, is_training=phase_flag, name='deconv2_1')
        print("deconv2_1:", deconv2_1.shape)#deconv2_1 (1, 24, 24, 24, 256) 这个家伙会把通道数量增加
        concat_2 = tf.concat([deconv2_1, conv3_1], axis=concat_dim, name='concat_2')
        print("concat_2 :", concat_2 .shape)#concat_2 (1, 24, 24, 24, 512)
        concat_2 = conv3d( concat_2, 128, kernel_size=1, stride=1, use_bias=False, name='concat_2_short' )
        deconv2_2 = clique_block(concat_2, channels_per_layer=128,outchannel=128, layer_num=layer_num, is_train=phase_flag,
                               keep_prob=keep_prob, block_name='b' + str(6))  # 注意输入的是卷积后的结果，输出也是卷积后的结果
        print("deconv2_2:", deconv2_2.shape)#deconv2_2 (1, 24, 24, 24, 128)

        deconv3_1 = deconv_bn_relu(input=deconv2_2, output_chn=128, is_training=phase_flag, name='deconv3_1')
        print("deconv3_1:", deconv3_1.shape)# deconv3_1 (1, 48, 48, 48, 128)
        concat_3 = tf.concat([deconv3_1, conv2_1], axis=concat_dim, name='concat_3')
        print("concat_3:", concat_3.shape)# concat_3(1, 48, 48, 48, 256)
        concat_3 = conv3d( concat_3, 64, kernel_size=1, stride=1, use_bias=False, name='concat_3_short' )
        deconv3_2 = clique_block(concat_3, channels_per_layer=64,outchannel=64, layer_num=layer_num, is_train=phase_flag,
                               keep_prob=keep_prob, block_name='b' + str(7))  # 注意输入的是卷积后的结果，输出也是卷积后的结果
        print("deconv3_2:",deconv3_2.shape)#deconv3_2 (1, 48, 48, 48, 64)

        deconv4_1 = deconv_bn_relu(input=deconv3_2, output_chn=64, is_training=phase_flag, name='deconv4_1')
        print("deconv4_1:", deconv4_1.shape)#deconv4_2 (1, 96, 96, 96, 64)
        concat_4 = tf.concat([deconv4_1, conv1_1], axis=concat_dim, name='concat_4')
        print(" concat_4:",  concat_4.shape)# deconv4_2 (1, 96, 96, 96,128)
        deconv4_2 = conv3d( concat_4, 32, kernel_size=1, stride=1, use_bias=False, name='concat_4_short' )
        # deconv4_2 = clique_block(concat_4, channels_per_layer=32, outchannel=32, layer_num=layer_num,
        #                          is_train=phase_flag,
        #                          keep_prob=keep_prob, block_name='b' + str(8))  # 注意输入的是卷积后的结果，输出也是卷积后的结果
        # print("deconv4_2:", deconv4_2.shape)  # deconv4_2 (1, 96, 96, 96, 32)
        deconv4_2_bn = tf.contrib.layers.batch_norm(deconv4_2, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=phase_flag, scope="batch_norm")
        deconv4_2_relu = tf.nn.relu(deconv4_2_bn, name='relu')
        pre_pro = conv3d(input=deconv4_2_relu, output_chn=self.output_chn, kernel_size=1, stride=1, use_bias=True, name='pre_pro')

        pred_prob= pre_pro#pred_prob (1, 96, 96, 96, 8) 注意在这里生成了最终预测

        # ======================用于预测输出=============================
        # auxiliary prediction 0
        aux0_conv = conv3d(input=deconv1_2, output_chn=self.output_chn, kernel_size=1, stride=1, use_bias=True,name='aux0_conv')#aux0_conv (1, 12, 12, 12, 8) 卷积成8类输出
        aux0_deconv_1 = Deconv3d(input=aux0_conv, output_chn=self.output_chn, name='aux0_deconv_1')#aux0_deconv_1 (1, 24, 24, 24, 8)
        aux0_deconv_2 = Deconv3d(input=aux0_deconv_1, output_chn=self.output_chn, name='aux0_deconv_2')#aux0_deconv_2 (1, 48, 48, 48, 8)
        aux0_prob = Deconv3d(input= aux0_deconv_2 , output_chn=self.output_chn, name='aux0_prob')#aux0_prob (1, 96, 96, 96, 8)

        # auxiliary prediction 1
        aux1_conv = conv3d(input=deconv2_2, output_chn=self.output_chn, kernel_size=1, stride=1, use_bias=True,name='aux1_conv')#aux1_conv (1, 24, 24, 24, 8)
        aux1_deconv_1 = Deconv3d(input=aux1_conv, output_chn=self.output_chn, name='aux1_deconv_1')# aux1_deconv_1 (1, 48, 48, 48, 8)
        aux1_prob = Deconv3d(input=aux1_deconv_1, output_chn=self.output_chn, name='aux1_prob')#aux1_prob (1, 96, 96, 96, 8)

        # auxiliary prediction 2
        aux2_conv = conv3d(input=deconv3_2, output_chn=self.output_chn, kernel_size=1, stride=1, use_bias=True,name='aux2_conv')# aux2_conv (1, 48, 48, 48, 8)
        aux2_prob = Deconv3d(input=aux2_conv, output_chn=self.output_chn, name='aux2_prob')#aux2_prob (1, 96, 96, 96, 8)

        soft_prob = tf.nn.softmax(pred_prob, name='pred_soft')
        pred_label = tf.argmax(soft_prob, axis=4, name='argmax')

        return pred_prob, pred_label, aux0_prob, aux1_prob, aux2_prob

    # train function
    def train(self):
        """选中优化方法"""

        u_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.beta1).minimize(self.total_loss, var_list=self.u_vars)

        # 对网络参数进行初始化
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        # 初始化C3D model对应的权重  也就是conv1，conv2，conv3a，conv3b，conv4a和conv4b的权重，用作初始化
        self.initialize_finetune()

        # save .log
        #self.log_writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 1
        if self.load_chkpoint(self.chkpoint_dir,self.step):
            print(" [*] 加载checkpoint成功..")
        else:
            print(" [!] 未加载checkpoint...")

        # load all volume files
        pair_list = glob('{}/*.nii.gz'.format(self.traindata_dir))
        pair_list.sort()
       
        for i in range(len(pair_list)):
            print(str(pair_list[i]))

        #将体积转换后的体数据和标签分别放入对应的文件下面,里面每一个数据元素对应和
        img_clec, label_clec = load_data_pairs(pair_list, self.resize_r, self.rename_map)

        # temporary file to save loss
        loss_log = open("loss.txt", "w")
        self.sess.graph.finalize()  # 锁定图，使之只能读不能写，避免后面添加节点导致出错
        for epoch in np.arange(self.epoch):

            start_time = time.time()
            # 获取训练数据 其实这是随机去体数据中裁剪

            batch_img, batch_label = get_batch_patches(img_clec, label_clec, self.inputI_size, self.batch_size, chn=1, flip_flag=True, rot_flag=True)
            # 获取验证数据
            batch_val_img, batch_val_label = get_batch_patches(img_clec, label_clec, self.inputI_size, self.batch_size, chn=1, flip_flag=True, rot_flag=True)
            #获取训练路径掩码
            is_global_path, global_path_list, local_path_list=get_test_path_list(self.Stages,self.Blocks,self.Columns)

            # Update 3D U-net 获取损失值
            _, cur_train_loss = self.sess.run([u_optimizer, self.total_loss], feed_dict={self.input_I: batch_img,
                                                self.input_gt: batch_label,self.is_global_path:is_global_path,
                                              self.global_path_list:global_path_list,self.local_path_list:local_path_list})
            # self.log_writer.add_summary(summary_str, counter)

            # #取出diceloss,参见https://blog.csdn.net/yeqiang19910412/article/details/78651939
            # cur_valid_loss = self.total_loss.eval({self.input_I: batch_val_img, self.input_gt: batch_val_label})
            # #获取该图所对应的label
            # cube_label = self.sess.run(self.pred_label, feed_dict={self.input_I: batch_val_img})
            # # 计算dice值
            # dice_c = []
            # for c in range(self.output_chn):
            #     ints = np.sum(((batch_val_label[0,:,:,:]==c)*1)*((cube_label[0,:,:,:]==c)*1))
            #     union = np.sum(((batch_val_label[0,:,:,:]==c)*1) + ((cube_label[0,:,:,:]==c)*1)) + 0.0001
            #     dice_c.append((2.0*ints)/union)
            # dice_c = np.around( dice_c, decimals=3 )
            # print(dice_c)
            #
            counter += 1
            print("Epoch: [%2d] ：....time: %4.4f........................train_loss: %.8f" % (epoch, time.time() - start_time, cur_train_loss))
            # #打印当前训练值
            # loss_log.write( "%s   %s    %s   %s\n" % (counter,cur_train_loss, cur_valid_loss, dice_c) )  # 把loss给保留下来
            if np.mod(counter, self.save_intval) == 0:#隔着多少个epoch采取一次保存策略

                self.test(counter,"train.log")
                self.save_chkpoint( self.chkpoint_dir, self.model_name, counter )
        loss_log.close()

    # 测试测试集中jaccard指数
    def test(self,counter,logname):

        test_log = open( logname, "a" )

        # 获得test数据列表
        test_list = glob('{}/*.nii.gz'.format(self.testdata_dir))
        test_list.sort()

        # 用于保存label数据集中所有文件的数量
        all_dice = np.zeros([int(len(test_list)/2), 8])
        all_jaccard = np.zeros( [int( len( test_list ) / 2 ), 8] )
        # test
        for k in range(0, len(test_list), 2):
            print("开始处理:",str(test_list[k]))

            # 加载体数据 这里是加载原始数据
            vol_file = nib.load(test_list[k])

            # 将体数据转换为矩阵
            vol_data = vol_file.get_data().copy()

            #尺度缩放到307
            resize_dim = (np.array(vol_data.shape) * self.resize_r).astype('int')
            vol_data_resz = resize(vol_data, resize_dim, order=1, preserve_range=True)

            # 对体数据标准化
            vol_data_resz = vol_data_resz.astype('float32')
            vol_data_resz = vol_data_resz / 255.0

            # 将体数据分解为单个立方块，用于进行预测
            cube_list = decompose_vol2cube(vol_data_resz, self.batch_size, self.inputI_size, self.inputI_chn, self.ovlp_ita)

            # 获取预测出来的label
            cube_label_list = []
            for c in range(len(cube_list)):
                #取出一个立方块 并且进行标准化
                cube2test = cube_list[c]
                mean_temp = np.mean(cube2test)
                dev_temp = np.std(cube2test)
                cube2test_norm = (cube2test - mean_temp) / dev_temp
                if c%20==0:
                    print("预测第%s个ct图像第%s立方块"%(k,c))
                #获取单个立方块的预测结果
                # 获取测试路径掩码
                is_global_path, global_path_list, local_path_list = get_test_path_list(self.Stages, self.Blocks,
                                                                                        self.Columns)
                self.sess.graph.finalize()
                cube_label = self.sess.run(self.pred_label, feed_dict={self.input_I: cube2test_norm,self.is_global_path:is_global_path,
                                              self.global_path_list:global_path_list,self.local_path_list:local_path_list})
                cube_label_list.append(cube_label)

            # 将这些立方块的结果拼凑起来
            composed_orig = compose_label_cube2vol(cube_label_list, resize_dim, self.inputI_size, self.ovlp_ita, self.output_chn)
            composed_label = np.zeros(composed_orig.shape, dtype='int16')

            # 对label进行重命名，也就是将标签值转换回0, 205, 420, 500, 550, 600, 820, 850
            for i in range(len(self.rename_map)):
                composed_label[composed_orig == i] = self.rename_map[i]
            composed_label = composed_label.astype('int16')


            # resize回512的原始体数据大小
            composed_label_resz = resize(composed_label, vol_data.shape, order=0, preserve_range=True)
            composed_label_resz = composed_label_resz.astype('int16')

            # 加载对应的groundtruth
            gt_file = nib.load(test_list[k + 1])
            #将标签体数据转换成矩阵
            gt_label = gt_file.get_data().copy()

            k_dice_c = seg_eval_metric(composed_label_resz, gt_label)#计算n分类dice指数
            print ("dice为:",k_dice_c)
            all_dice[int(k/2), :] = np.asarray(k_dice_c)

            #############################################
            k_jaccard_c = jaccard_n_class( composed_label_resz, gt_label )  # 计算n分类jaccard指数
            print( "jaccard为:", k_jaccard_c )
            all_jaccard[int( k / 2 ), :] = np.asarray( k_jaccard_c )
            #######################################

        mean_dice = np.mean(all_dice, axis=0)#计算多个体数据平均jaccard指数
        mean_dice = np.around( mean_dice, decimals=3 )
        print ("average dice: ")
        print (mean_dice)

        #########################################
        mean_jaccard = np.mean(all_jaccard , axis=0)#计算多个体数据平均jaccard指数
        mean_jaccard  = np.around( mean_jaccard, decimals=3 )
        print ("average jaccard: ")
        print (mean_jaccard)
        #########################################

        total_mean_dice=0
        total_mean_jaccard = 0
        for i in range(1,8):#背景类不计入
            total_mean_dice+=mean_dice[i]
            total_mean_jaccard += mean_jaccard[i]
        test_log.write( "Epoch %s average dice:  %s dicemean: %s average jaccard:  %s jaccardmean: %s\n"\
                        % ( counter,str( mean_dice ),str(total_mean_dice/7),str( mean_jaccard ),str(total_mean_jaccard/7)) )
        test_log.close()
    # test function for cross validation
    def test4crsv(self):

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        print("尝试加载保存点于:",self.chkpoint_dir,self.step)
        start_time = time.time()
        if self.load_chkpoint(self.chkpoint_dir,self.step):
            print(" [*] 加载成功")
        else:
            print(" [!] 加载失败...")
            return
        self.test(self.step, "test.log" )

    def generate_map(self,counter):
        # 获得test数据列表
        test_list = glob('{}/*.nii.gz'.format(self.testdata_dir))
        test_list.sort()

        # test
        for k in range(0, len(test_list),1):
            print("开始处理:", str(test_list[k]))

            # 加载体数据 这里是加载原始数据
            vol_file = nib.load(test_list[k])
            ref_affine=vol_file.affine
            vol_data = vol_file.get_data().copy()

            # # ====================
            # if k == 22 or k == 36:
            #     # flip
            #     vol_data = vol_data[::-1, :, :]
            # # ====================

            # 尺度缩放到307
            resize_dim = (np.array(vol_data.shape) * self.resize_r).astype('int')
            vol_data_resz = resize(vol_data, resize_dim, order=1, preserve_range=True)

            # 对体数据标准化
            vol_data_resz = vol_data_resz.astype('float32')
            vol_data_resz = vol_data_resz / 255.0

            # 将体数据分解为单个立方块，用于进行预测
            cube_list = decompose_vol2cube(vol_data_resz, self.batch_size, self.inputI_size, self.inputI_chn,
                                           self.ovlp_ita)
            print(k," cube list is built!")
            # 获取预测出来的label
            cube_label_list = []
            for c in range(len(cube_list)):
                # 取出一个立方块 并且进行标准化
                cube2test = cube_list[c]
                mean_temp = np.mean(cube2test)
                dev_temp = np.std(cube2test)
                cube2test_norm = (cube2test - mean_temp) / dev_temp
                if c % 20 == 0:
                    print("预测第%s个ct图像第%s立方块" % (k, c))
                # 获取单个立方块的预测结果
                # 获取测试路径掩码
                is_global_path, global_path_list, local_path_list = get_test_path_list(self.Stages, self.Blocks,
                                                                                       self.Columns)
                cube_label = self.sess.run(self.pred_label,
                                           feed_dict={self.input_I: cube2test_norm, self.is_global_path: is_global_path,
                                                      self.global_path_list: global_path_list,
                                                      self.local_path_list: local_path_list})
                cube_label_list.append(cube_label)

            # 将这些立方块的结果拼凑起来
            composed_orig = compose_label_cube2vol(cube_label_list, resize_dim, self.inputI_size, self.ovlp_ita,self.output_chn)
            composed_label = np.zeros(composed_orig.shape, dtype='int16')
            print("=== prob volume is composed!")

            # 对label进行重命名，也就是将标签值转换回0, 205, 420, 500, 550, 600, 820, 850
            for i in range(len(self.rename_map)):
                composed_label[composed_orig == i] = self.rename_map[i]
            composed_label = composed_label.astype('int16')

            #
            composed_label_resz = resize(composed_label, vol_data.shape, order=0, preserve_range=True)
            composed_label_resz = composed_label_resz.astype('int16')

            composed_label_resz = remove_minor_cc( composed_label_resz, rej_ratio=0.3, rename_map=self.rename_map )
            # # ====================
            # if k == 22 or k == 36:
            #     # flip
            #     composed_label_resz = composed_label_resz[::-1, :, :]
            # # ====================
            c_map_path = os.path.join(self.labeling_dir, ('ct_test_' + str(2001 + k)+ '_image.nii.gz'))
            labeling_vol=nib.Nifti1Image(composed_label_resz, ref_affine)
            nib.save(labeling_vol, c_map_path)



    # 把预测出来的结果保存起来
    def test_generate_map(self):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        print("尝试加载保存点于",self.chkpoint_dir,self.step)
        start_time = time.time()
        if self.load_chkpoint(self.chkpoint_dir,self.step):
            print(" [*] 加载成功")
        else:
            print(" [!] 加载失败...")
            return
        self.generate_map(self.step )


    # save checkpoint file
    def save_chkpoint(self, checkpoint_dir, model_name, step):
        model_dir = "%s_%s_%s" % (self.batch_size, self.outputI_size,step)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    # load checkpoint file
    def load_chkpoint(self, checkpoint_dir,step=-1):

        tf.reset_default_graph()
        model_dir = "%s_%s_%s" % (self.batch_size, self.outputI_size,step)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        print( " [*] 加载保存点文件,路径为",str(checkpoint_dir) )
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)#在对应dir查找检查点文件
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    # load C3D model
    def initialize_finetune(self):
        checkpoint_dir = '../outcome/model/C3D_unet_1chn'
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver_ft.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print("fine-turn成功！")
