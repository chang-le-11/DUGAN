import tensorflow as tf
WEIGHT_INIT_STDDEV = 0.05

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)

def conv_lrelu_block(x,kernel,use_lrelu=True,Scope=None,BN=False):
    x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode = 'REFLECT')
    out = tf.nn.conv2d(x_padded, kernel, strides=[1, 1, 1, 1], padding='VALID')
    if BN:
        with tf.variable_scope(Scope):
            out = tf.layers.batch_normalization(out, training=True)
    if use_lrelu:
        out = lrelu(out)
    return out


#
def gradient(img):
	kernel = tf.constant([[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]])
	kernel = tf.expand_dims(kernel, axis = -1)
	kernel = tf.expand_dims(kernel, axis = -1)
	grad_img = tf.nn.conv2d(img, kernel, strides = [1, 1, 1, 1], padding = 'SAME')
	return grad_img


# def gradient(input):
#     filter1 = tf.reshape(tf.constant([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]), [3, 3, 1, 1])
#     filter2 = tf.reshape(tf.constant([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]), [3, 3, 1, 1])
#     Gradient1 = tf.nn.conv2d(input, filter1, strides=[1, 1, 1, 1], padding='SAME')
#     Gradient2 = tf.nn.conv2d(input, filter2, strides=[1, 1, 1, 1], padding='SAME')
#     Gradient = tf.abs(Gradient1) + tf.abs(Gradient2)
#     return Gradient


#
class Generator(object):

    def __init__(self, scope_name):
        self.scope = scope_name
        with tf.variable_scope('Generator'):
            self.conv1 = self._create_variables(2, 16, 3, scope='conv1')
            self.conv2 = self._create_variables(16, 16, 3, scope='conv2')
            self.conv3 = self._create_variables(32, 16, 3, scope='conv3')

            self.conv4 = self._create_variables(48, 16, 3, scope='conv4')

            self.conv1_grad = self._create_variables(1, 16, 3, scope='conv1_grad')
            self.conv2_grad = self._create_variables(32, 16, 3, scope='conv2_grad')
            self.conv3_grad = self._create_variables(48, 16, 3, scope='conv3_grad')
            self.conv4_grad = self._create_variables(64, 16, 3, scope='conv4_grad')

            self.grad_out = self._create_variables(64, 1, 3, scope='grad_out')

            # nodis, traddis, FF, att, gra
            #self.grad_out = self._create_variables(16, 1, 3, scope='grad_out')

            self.de1 = self._create_variables(128, 64, 3, scope='de1')
            self.de2 = self._create_variables(64, 32, 3, scope='de2')
            self.de3 = self._create_variables(32, 16, 3, scope='de3')
            self.de4 = self._create_variables(16, 1, 3, scope='de4')



    # 创建变量函数
    def _create_variables(self, input_filters, output_filters, kernel_size, scope):
        # tf.variable_scope()创建一个变量空间，用于参数共享
        with tf.variable_scope(scope):
            shape = [kernel_size, kernel_size, input_filters, output_filters]
            # tf.get_variable()创建变量，会先搜索变量的名称，有就直接用，没有再创建
            kernel = tf.get_variable('kernel', shape=shape,
                                     initializer=tf.truncated_normal_initializer(stddev=WEIGHT_INIT_STDDEV))
        return kernel

    def transform(self, vis, ir):
        x = tf.concat([vis, ir], 3)
        vi_grad = gradient(vis)
        ir_grad = gradient(ir)
        grad = tf.maximum(vi_grad, ir_grad)
        conv1 = conv_lrelu_block(x, kernel = self.conv1, use_lrelu=True, Scope=self.scope+'conv1', BN=True)
        conv2 = conv_lrelu_block(conv1, kernel=self.conv2, use_lrelu=True, Scope=self.scope + 'conv2', BN=True)
        f1 = tf.concat([conv1, conv2], 3)
        conv3 = conv_lrelu_block(f1, kernel=self.conv3, use_lrelu=True, Scope=self.scope + 'conv3', BN=True)
        f2 = tf.concat([f1, conv3], 3)
        conv4 = conv_lrelu_block(f2, kernel=self.conv4, use_lrelu=True, Scope=self.scope + 'conv4', BN=True)
        conv1_grad = conv_lrelu_block(grad, kernel=self.conv1_grad, use_lrelu=True, Scope=self.scope + 'conv1_grad', BN=True)
        f1_grad = tf.concat([conv1, conv1_grad], 3)
        conv2_grad = conv_lrelu_block(f1_grad, kernel=self.conv2_grad, use_lrelu=True, Scope=self.scope + 'conv2_grad', BN=True)
        f2_grad = tf.concat([conv1_grad, conv2_grad, conv2], 3)
        conv3_grad = conv_lrelu_block(f2_grad, kernel=self.conv3_grad, use_lrelu=True, Scope=self.scope + 'conv3_grad', BN=True)
        f3_grad = tf.concat([conv3_grad, conv1_grad, conv2_grad, conv3], 3)
        conv4_grad = conv_lrelu_block(f3_grad, kernel=self.conv4_grad, use_lrelu=True, Scope=self.scope + 'conv4_grad', BN=True)
        f4_grad = tf.concat([conv3_grad, conv1_grad, conv2_grad, conv4_grad], 3)
        grad_out = conv_lrelu_block(f4_grad, kernel=self.grad_out, use_lrelu=False, Scope=self.scope+'grad_out', BN=False)

        grad_out = tf.nn.tanh(grad_out) / 2 + 0.5

        img = tf.concat([f2, conv4, f4_grad], 3)
        # img = tf.concat([f2, conv4], 3)
        de1 = conv_lrelu_block(img, kernel=self.de1, use_lrelu=True, Scope=self.scope + 'de1', BN=True)
        de2 = conv_lrelu_block(de1, kernel=self.de2, use_lrelu=True, Scope=self.scope + 'de2', BN=True)
        de3 = conv_lrelu_block(de2, kernel= self.de3, use_lrelu=True, Scope=self.scope + 'de3', BN=True)
        de4 = conv_lrelu_block(de3, kernel=self.de4, use_lrelu=False, Scope=self.scope+'de4', BN=False)
        de4 = tf.nn.tanh(de4) / 2 + 0.5

        return grad_out, de4
