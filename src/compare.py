#####################################
#判断两个体数据是否相等
#####################################
from __future__ import division
from ops import *
from utils import *
from seg_eval import *


#gt数据
testlist1="../outcome/testdata/ct_train_1004_label.nii.gz"
#预测出来的数据
testlist2="../outcome/label/ct_test_2001_image.nii.gz"

print(testlist1)
# 加载对应的groundtruth
gt_file = nib.load(testlist1)
# 将标签体数据转换成矩阵
gt_label = gt_file.get_data().copy()
print("标签数据加载完毕")

# 加载对应的groundtruth
predict_file = nib.load(testlist2)
# 将标签体数据转换成矩阵
predict_label = predict_file.get_data().copy()
print("预测数据加载完毕")

k_dice_c = seg_eval_metric(predict_label, gt_label)#计算n分类dice指数
print ("dice为:",k_dice_c)
k_jaccard_c = jaccard_n_class(predict_label, gt_label)  # 计算n分类jaccard指数
print("jaccard为:", k_jaccard_c)