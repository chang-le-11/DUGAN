#coding:utf-8
from __future__ import print_function
import time
import numpy as np
from generate import generate
import os
import glob
import tensorflow as tf
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)

# 批量大小
BATCH_SIZE =16
# 训练周期
EPOCHES = 20

LOGGING = 50
MODEL_SAVE_PATH = '../models/our/'
# 是否训练
IS_TRAINING = False

# 获取图像
def prepare_data_path(dataset_path):
    # os.listdir()获取路径下包含文件或文件夹名称的列表
    filenames = os.listdir(dataset_path)
    # os.getcwd()
    # data_dir = os.path.join(os.getcwd(), dataset_path)
    data_dir = dataset_path
    # 获取路径下后缀名为bmp,tif,jpg,png的所有图像。
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    data.extend(glob.glob((os.path.join(data_dir, "*.jpg"))))
    data.extend(glob.glob((os.path.join(data_dir, "*.png"))))
    # data.sort(key=lambda x: int(x[len(data_dir) + 1:-4]))
    # 按序号进行排列
    data.sort()
    filenames.sort()
    return data, filenames

def main():
    print('\nBegin to generate pictures ...\n')
    Time=[]
    # 模型保存的路径
    model_path = MODEL_SAVE_PATH+'4500.ckpt'
    # 生成图像要保存的位置
    save_path= '../result/1/'
    if not os.path.exists(save_path):
        os.makedirs((save_path))
    # os.makedirs(save_path)
    # 获取红外和可见光测试图像
    ir_paths,ir_names=prepare_data_path(r'../test_imgs/TNO/ir')
    vis_paths,vis_names=prepare_data_path(r'../test_imgs/TNO/vis')
    # 获取测试图像的数量
    for i in range(len(vis_paths)):
        # 读取每一幅测试图像
        ir_path = ir_paths[i]
        vis_path = vis_paths[i]
        # 开始时间
        begin = time.time()
        # 生成测试图像
        generate(ir_path, vis_path, model_path,vis_names[i], output_path= save_path)
        # 结束时间
        end = time.time()
        # 花费的时间
        Time.append(end - begin)
        print(vis_names[i])
    print("Time: mean:%s, std: %s" % (np.mean(Time)*1000, np.std(Time)))

if __name__ == '__main__':
	main()
