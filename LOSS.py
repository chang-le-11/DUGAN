import tensorflow as tf
import numpy as np



def _tf_fspecial_gauss(size, sigma):
	"""Function to mimic the 'fspecial' gaussian MATLAB function
	"""
	x_data, y_data = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]

	x_data = np.expand_dims(x_data, axis = -1)
	x_data = np.expand_dims(x_data, axis = -1)

	y_data = np.expand_dims(y_data, axis = -1)
	y_data = np.expand_dims(y_data, axis = -1)

	x = tf.constant(x_data, dtype = tf.float32)
	y = tf.constant(y_data, dtype = tf.float32)

	g = tf.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
	return g / tf.reduce_sum(g)

# L1损失函数，向量中各元素的绝对值之和
def L1_LOSS(batchimg):
	# input(B H W)
	L1_norm = tf.reduce_sum(tf.abs(batchimg), axis = [1, 2])
	# tf.norm(batchimg, axis = [1, 2], ord = 1) / int(batchimg.shape[1])
	# (B 1)
	E = tf.reduce_mean(L1_norm)
	# (1)
	return E

def Per_LOSS(batchimg):
	_, h, w, c = batchimg.get_shape().as_list()
	fro_2_norm = tf.reduce_sum(tf.square(batchimg),axis=[1,2,3])
	loss=fro_2_norm / (h * w * c)
	return loss

# 矩阵F范数，矩阵中各元素的平方和再开平方
def Fro_LOSS(batchimg):
	# (B H W)
	fro_norm = tf.square(tf.norm(batchimg, axis = [1, 2], ord = 'fro'))
	E = tf.reduce_mean(fro_norm)
	return E

def features_grad(features):
	kernel = tf.constant([[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]])
	kernel = tf.expand_dims(kernel, axis = -1)
	kernel = tf.expand_dims(kernel, axis = -1)
	_, _, _, c = features.shape
	c = int(c)
	for i in range(c):
		fg = tf.nn.conv2d(tf.expand_dims(features[:, :, :, i], axis = -1), kernel, strides = [1, 1, 1, 1],
		                  padding = 'SAME')
		if i == 0:
			fgs = fg
		else:
			fgs = tf.concat([fgs, fg], axis = -1)
	return fgs




def SSIM_LOSS(img1, img2, size=11, sigma=1.5):
    window = _tf_fspecial_gauss(size, sigma) # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1],padding='VALID')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='VALID') - mu1_mu2

    value = (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2)
    value = tf.reduce_mean(value)
    return value

def grad(img):
    kernel = tf.constant([[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]])
    kernel = tf.expand_dims(kernel, axis=-1)
    kernel = tf.expand_dims(kernel, axis=-1)
    grad_img = tf.nn.conv2d(img, kernel, strides=[1, 1, 1, 1], padding='SAME')
    return grad_img