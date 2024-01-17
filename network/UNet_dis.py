import tensorflow as tf
WEIGHT_INIT_STDDEV = 0.1
from tensorflow.keras.layers import Conv2D,Activation,add,multiply


def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)

def conv_lrelu_nopad_block(x,kernel,use_lrelu=True,Scope=None,BN=None):
    # 对输入图像进行填充
    # x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode = 'REFLECT')
    # 卷积过程
    out = tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='VALID')
    # 是否使用BN
    if BN:
        with tf.compat.v1.variable_scope(Scope):
            out = tf.compat.v1.layers.batch_normalization(out, training=True)
    # 是否使用relu激活函数
    if use_lrelu:
        out = lrelu(out)
    return out


def conv2d_2(x, kernel, bias, strides, use_relu = True, use_BN = True, Scope = None, Reuse = None):
	x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode = 'REFLECT')
	out = tf.nn.conv2d(x_padded, kernel, strides, padding = 'VALID')
	out = tf.nn.bias_add(out, bias)
	if use_BN:
		with tf.compat.v1.variable_scope(Scope):
			out = tf.compat.v1.layers.batch_normalization(out, training = True, reuse = Reuse)
	if use_relu:
		out = lrelu(out)
	return out

def l2_norm(input_x, epsilon=1e-12):
    input_x_norm = input_x / (tf.reduce_sum(input_x ** 2) ** 0.5 + epsilon)
    return input_x_norm


def hw_flatten(x):
	return tf.layers.flatten(x)



def attention_block2(x, g, inter_channel, data_format='channels_last', scope_name=None):
	with tf.variable_scope(scope_name):
    # theta_x(?,g_height,g_width,inter_channel)
		theta_x = Conv2D(inter_channel, [1, 1], strides=[1, 1], data_format=data_format)(x)
		# phi_g(?,g_height,g_width,inter_channel)
		phi_g = Conv2D(inter_channel, [1, 1], strides=[1, 1], data_format=data_format)(g)
		# f(?,g_height,g_width,inter_channel)
		f = Activation('relu')(add([theta_x, phi_g]))
		# psi_f(?,g_height,g_width,1)
		psi_f = Conv2D(1, [1, 1], strides=[1, 1], data_format=data_format)(f)
		rate = Activation('sigmoid')(psi_f)
		# rate(?,x_height,x_width)
		# att_x(?,x_height,x_width,x_channel)
		att_x = multiply([x, rate])
	return att_x



def upsample(inputs, shape):
    out = tf.compat.v1.image.resize_bilinear(images=inputs,size=[shape[1],shape[2]])
    return out


class U_NetDiscriminator(object):
	def __init__(self, scope_name):
		self.weight_vars = []
		self.scope = scope_name

		# tf.variable_scope()创建一个变量空间，用于参数共享
		with tf.variable_scope(scope_name):
			# 输入通道1，输出通道16，卷积核大小3×3
			self.first_conv1, self.first_bias1 = self._create_variables(1, 32, 3, scope='frist1')
			self.kernel_conv1, self.bias_conv1 = self._create_variables(32, 64, 3, scope='conv1')

			self.kernel_conv2, self.bias_conv2 = self._create_variables(64, 128, 3, scope='conv2')
			self.kernel_conv3, self.bias_conv3 = self._create_variables(128, 256, 3, scope='conv3')

			self.h1_to_d1_conv, self.h1_to_d1_bias = self._create_variables(32, 32, 3, scope='h1')

			self.out4_out3_conv, self.out4_out3_bias = self._create_variables(256, 128, 3, scope='up1')
			self.out3_out2_conv, self.out3_out2_bias = self._create_variables(256, 128, 3, scope='up2')

			self.up1_conv, self.up1_bias = self._create_variables(128, 64, 3, scope='up3')
			self.up2_conv, self.up2_bias = self._create_variables(128, 64, 3, scope='up4')

			self.up3_conv, self.up3_bias = self._create_variables(64, 32, 3, scope='up5')
			self.up4_conv, self.up4_bias = self._create_variables(64, 32, 3, scope='up6')

			self.conv5, self.bias5 = self._create_variables(32, 32, 3, scope='conv5')
			self.conv6, self.bias6 = self._create_variables(32, 1, 3, scope='conv6')

			self.gated, self.biasg = self._create_variables(256, 128, 1, scope='gated')



	def _create_variables(self, input_filters, output_filters, kernel_size, scope):
		shape = [kernel_size, kernel_size, input_filters, output_filters]
		with tf.variable_scope(scope):
			# tf.get_variable()创建变量，会先搜索变量的名称，有就直接用，没有再创建
			# tf.truncated_normal_initializer()从截断的正态分布中输出随机数
			kernel = tf.get_variable('kernel', shape=shape,
									 initializer=tf.truncated_normal_initializer(stddev=WEIGHT_INIT_STDDEV))
			bias =tf.get_variable('bias',shape=[output_filters],initializer=tf.zeros_initializer())
		return (kernel, bias)

	def discrim(self, img, reuse):
		if len(img.shape) != 4:
			img = tf.expand_dims(img, -1)
		out_img = img
		out1 = conv2d_2(out_img, self.first_conv1, self.first_bias1, [1,1,1,1], use_relu=True, use_BN=False, Scope=self.scope+'./first', Reuse=reuse)
		out2 = conv2d_2(out1, self.kernel_conv1, self.bias_conv1, [1, 2, 2, 1], use_relu=True, use_BN=True, Scope=self.scope + '/bn1' , Reuse=reuse)
		out3 = conv2d_2(out2, self.kernel_conv2, self.bias_conv2, [1, 2, 2, 1], use_relu=True, use_BN=True,
					   Scope=self.scope + '/bn2', Reuse=reuse)
		out4 = conv2d_2(out3, self.kernel_conv3, self.bias_conv3, [1, 2, 2, 1], use_relu=True, use_BN=True,
					   Scope=self.scope + '/bn3', Reuse=reuse)
		gated = conv2d_2(out4, self.gated, self.biasg, [1, 1, 1, 1], use_relu=True, use_BN=False,
						 Scope=self.scope + './gated', Reuse=reuse)  # 128 channel
		shape = gated.get_shape().as_list()
		out = tf.layers.average_pooling2d(inputs=gated, pool_size=shape[1], strides=1, padding='VALID',
										  name='global_vaerage_pool')
		out_flatten = tf.reshape(out, shape=[-1, int(out.shape[1]) * int(out.shape[2]) * int(out.shape[3])])
		out_logit = tf.layers.dense(out_flatten, 1, activation=None, use_bias=True, trainable=True, reuse=reuse)
		out_activate = tf.nn.tanh(out_logit) / 2 + 0.5

		gated1 = upsample(gated, shape=tf.shape(out3))
		gated2 = upsample(gated, shape=tf.shape(out2))
		gated3 = upsample(gated, shape=tf.shape(out1))

		att3 = attention_block2(out3, gated1, inter_channel=128, scope_name='att33')
		att2 = attention_block2(out2, gated2, inter_channel=64, scope_name='att22')
		att1 = attention_block2(out1, gated3, inter_channel=32, scope_name='att11')

		out4_up = upsample(out4, shape=tf.shape(out3))
		out4_up = conv2d_2(out4_up, self.out4_out3_conv, self.out4_out3_bias, [1, 1, 1, 1], use_relu=True, use_BN=True,
										 Scope=self.scope+'/up1', Reuse=reuse)
		up1 = tf.concat([att3, out4_up], 3)
		up1_conv = conv2d_2(up1, self.out3_out2_conv, self.out3_out2_bias, [1, 1, 1, 1], use_relu=True, use_BN=True,
										   Scope=self.scope+'./up2', Reuse=reuse)
		up1_up = upsample(up1_conv, shape=tf.shape(out2))
		up1_up = conv2d_2(up1_up, self.up1_conv, self.up1_bias, [1, 1, 1, 1], use_relu=True, use_BN=True,
										Scope=self.scope+'./up3', Reuse=reuse)
		up2 = tf.concat([att2, up1_up], 3)
		up2_conv = conv2d_2(up2, self.up2_conv, self.up2_bias, [1, 1, 1, 1], use_relu=True, use_BN=True,
										  Scope=self.scope+'./up4', Reuse=reuse)
		up2_up = upsample(up2_conv, shape=tf.shape(out1))
		up2_up = conv2d_2(up2_up, self.up3_conv, self.up3_bias, [1, 1,1,1], use_relu=True, use_BN=True,
										Scope=self.scope+'./up5', Reuse=reuse)
		up3 = tf.concat([att1, up2_up], 3)
		up3_conv = conv2d_2(up3, self.up4_conv, self.up4_bias, [1,1,1,1], use_relu=True, use_BN=True,
										  Scope=self.scope+'/up6', Reuse=reuse)

		output = conv2d_2(up3_conv, self.conv5, self.bias5, [1, 1, 1, 1], use_relu=True, use_BN=True,
										Scope=self.scope+'conv6', Reuse=reuse)
		output = conv2d_2(output, self.conv6, self.bias6, [1,1,1,1], use_relu=False, use_BN=False,
						  Scope=self.scope+'./conv6', Reuse=reuse)
		# print(output.shape)
		output = tf.nn.tanh(output) / 2 + 0.5
		# print(output.shape)
		return out_activate, output

