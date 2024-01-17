#coding:utf-8
from __future__ import print_function
import time
import h5py
import numpy as np
from train import train
from Test.generate import generate
import os 
import glob
import tensorflow as tf


# 批量大小，每次训练从训练集中取16个样本
BATCH_SIZE =16
# 训练周期，一个epoch相当于使用训练集中全部样本训练一遍
EPOCHES = 10

LOGGING = 50
# 模型保存路径
MODEL_SAVE_PATH = 'models/model_5.8/'
# 训练还是测试
IS_TRAINING = True


# 读取路径下的图像
def prepare_data_path(dataset_path):
	# os.listdir()返回路径下包含文件或文件夹名称的列表
    filenames = os.listdir(dataset_path)
    # os.getcwd() 
    # data_dir = os.path.join(os.getcwd(), dataset_path)
    data_dir = dataset_path
	# 获取路径下后缀名为bmp、tif、jpg和png的所有图片文件
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
	# 判断是否训练
	if IS_TRAINING:
		# 打开路径下的H5格式的文件
		f = h5py.File('./train.h5', 'r')
		sources = f['data'][:]
		# np.transpose()对矩阵进行转置，例如原维度为(1,256,256,1),转置后的维度为(1,256,1,256)
		sources = np.transpose(sources, (0, 2, 3, 1))
		print(('\nBegin to train the network ...\n'))
		# 开始训练
		train(sources, MODEL_SAVE_PATH, EPOCHES, BATCH_SIZE, logging_period = LOGGING)

	# 开始测试
	else:
		print('\nBegin to generate pictures ...\n')
		Time=[]
		# 模型保存的路径
		model_path = MODEL_SAVE_PATH+'model.ckpt'
		# 生成图像要保存的位置
		save_path= 'result/results/'
		# os.makedirs(save_path)
		# 获取红外和可见光测试图像
		ir_paths,ir_names=prepare_data_path(r'./test_imgs/ir')
		vis_paths,vis_names=prepare_data_path(r'./test_imgs/vis')
		# 获取测试图像的数量
		for i in range(len(ir_paths)):
			# 读取每一幅测试图像
			ir_path = ir_paths[i]
			vis_path = vis_paths[i]
			# 开始时间
			begin = time.time()
			# 生成图像
			generate(ir_path, vis_path, model_path,ir_names[i], output_path= save_path)
			# 结束时间
			end = time.time()
			# 花费的时间
			Time.append(end - begin)
			print(ir_names[i])
		print("Time: mean:%s, std: %s" % (np.mean(Time), np.std(Time)))

if __name__ == '__main__':
	main()
