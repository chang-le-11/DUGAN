import tensorflow as tf
# from imageio import imread, imsave
from scipy.misc import imread, imsave
from network.Generator import Generator, gradient
import numpy
import numpy as np


# 生成图像
def generate(ir_path, vis_path, model_path, outputname, output_path = None):
	# 以灰度图的形式读取红外图像，并归一化
	ir_img = imread(ir_path, flatten=True)/ 255.0
	# 以灰度图的形式读取可见光图像并归一化
	vis_img = imread(vis_path, flatten=True)/ 255.0
	# 读取红外和可见光图像的维度
	ir_dimension = list(ir_img.shape)
	vis_dimension = list(vis_img.shape)
	# 默认读取的图像的维度为(H, W)
	# 在红外图像维度最前面插入一个维度1，表示一张测试图像
	ir_dimension.insert(0, 1)
	# # 在红外图像维度最后面加一个维度，表示单通道图像
	ir_dimension.append(1)
	vis_dimension.insert(0, 1)
	vis_dimension.append(1)
	# 在不改变数组内容的前提下，改变数组的形状。
	ir_img = ir_img.reshape(ir_dimension)
	vis_img = vis_img.reshape(vis_dimension)

	with tf.Graph().as_default(), tf.Session() as sess:
		# 输入的红外和可见光图像
		SOURCE_VIS = tf.placeholder(tf.float32, shape = vis_dimension, name = 'SOURCE_VIS')
		SOURCE_IR = tf.placeholder(tf.float32, shape = ir_dimension, name = 'SOURCE_IR')

		# 创建一个生成器实例
		G = Generator('Generator')
		# 输入红外和可见光图像，输出融合图像
		grad_out, fusion_img = G.transform(vis=SOURCE_VIS, ir= SOURCE_IR)
		theta_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
		# 保存训练过程中的参数
		saver = tf.train.Saver(theta_G)
		# 恢复模型路径下保存的变量
		saver.restore(sess, model_path)
		# var_list=saver._var_list
		# 运行，输出融合图像。feed_dict使用后面的值来替换前面的值
		output = sess.run(fusion_img, feed_dict = {SOURCE_VIS: vis_img, SOURCE_IR:ir_img})

		output = numpy.array(output)
		output = output[0, :, :, 0]
		# 保存图像，第一个参数是保存的路径，第二个参数是保存的内容
		imsave(output_path + outputname, output)

