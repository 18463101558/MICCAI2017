import os
import tensorflow as tf

from ini_file_io import load_train_ini
from model import unet_3D_xy
import os
# set cuda visable device
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def main(_):
    # load training parameter #
    ini_file = '../outcome/model/ini/tr_param.ini'
    param_sets = load_train_ini(ini_file)#获取参数信息
    param_set = param_sets[0]


    print('====== Phase >>> %s <<< ======' % param_set['phase'])#打印出属于训练阶段

    if not os.path.exists(param_set['chkpoint_dir']):#../outcome/model/checkpoint
        os.makedirs(param_set['chkpoint_dir'])
    if not os.path.exists(param_set['labeling_dir']):
        os.makedirs(param_set['labeling_dir'])#/home/xinyang/project_xy/mmwhs2017/dataset/common/hybrid_ct_map

    # GPU设置，per_process_gpu_memory_fraction表示95％GPU MEM，allow_growth表示不固定内存
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95, allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:
        model = unet_3D_xy(sess, param_set)#定义一个模型类，并且把参数传递给它，这个时候网络骨架搭建完毕
        if param_set['phase'] == 'train':
            try:
                os.remove( "train.log" )
            except:
                pass
            model.train()#进入训练阶段
        #elif param_set['phase'] == 'test':
        #   model.test_generate_map()
        elif param_set['phase'] == 'crsv':
            try:
                os.remove( "test.log" )
            except:
                pass
            model.test4crsv()

if __name__ == '__main__':
    tf.app.run()
