#!/usr/bin/python
import configparser

def load_train_ini(ini_file):
    # initialize 获取参数信息
    cf = configparser.ConfigParser()
    cf.read(ini_file)
    # dictionary list
    param_sections = []

    s = cf.sections()
    for d in range(len(s)):
        # create dictionary
        level_dict = dict(phase         = cf.get(s[d], "phase"),
                          batch_size    = cf.getint(s[d], "batch_size"),
                          inputI_size   = cf.getint(s[d], "inputI_size"),
                          inputI_chn    = cf.getint(s[d], "inputI_chn"),
                          outputI_size  = cf.getint(s[d], "outputI_size"),
                          output_chn    = cf.getint(s[d], "output_chn"),
                          rename_map    = cf.get(s[d], "rename_map"),
                          resize_r      = cf.getfloat(s[d], "resize_r"),
                          traindata_dir = cf.get(s[d], "traindata_dir"),
                          chkpoint_dir  = cf.get(s[d], "chkpoint_dir"),
                          learning_rate = cf.getfloat(s[d], "learning_rate"),
                          beta1         = cf.getfloat(s[d], "beta1"),
                          epoch         = cf.getint(s[d], "epoch"),
                          model_name    = cf.get(s[d], "model_name"),
                          save_intval   = cf.getint(s[d], "save_intval"),
                          testdata_dir  = cf.get(s[d], "testdata_dir"),
                          labeling_dir  = cf.get(s[d], "labeling_dir"),
                          ovlp_ita      = cf.getint(s[d], "ovlp_ita"),
                          step = cf.getint(s[d],"step"))
        # add to list
        param_sections.append(level_dict)

    return param_sections