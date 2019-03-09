import tensorflow as tf
from tensorflow.python.framework import ops
from libs.sn import spectral_normed_weight
from constants import SPECTRAL_NORM_UPDATE_OPS

class batch_norm(object):
    assigners = []
    shadow_variables = []

    """Code modification of http://stackoverflow.com/a/33950177"""
    def __init__(self, is_train, convolutional=True, decay=0.99, epsilon=1e-5, scale_after_normalization=True,
                 name="batch_norm"):
        with tf.variable_scope(name) as _:
            self.convolutional = convolutional
            self.is_train = is_train
            self.epsilon = epsilon
            self.ema = tf.train.ExponentialMovingAverage(decay=decay)
            self.scale_after_normalization = scale_after_normalization
            self.name=name

    def __call__(self, x):
        shape = x.get_shape().as_list()
        with tf.variable_scope(self.name) as _:
            depth = shape[-1]
            self.gamma = tf.get_variable("gamma", shape=[depth],
                                initializer=tf.random_normal_initializer(1., 0.02, seed=4285))
            self.beta = tf.get_variable("beta", shape=[depth],
                                initializer=tf.constant_initializer(0.))
            self.mean = tf.get_variable('mean', shape=[depth],
                                        initializer=tf.constant_initializer(0),
                                        trainable=False)
            self.variance = tf.get_variable('variance', shape=[depth],
                                        initializer=tf.constant_initializer(1),
                                        trainable=False)
            
            # Add to assigners if not already added previously.
            if not tf.get_variable_scope().reuse:
                batch_norm.assigners.append(self.ema.apply([self.mean, self.variance]))
                batch_norm.shadow_variables += [self.ema.average(self.mean), self.ema.average(self.variance)]

            if self.convolutional:
                x_unflattened = x
            else:
                x_unflattened = tf.reshape(x, [-1, 1, 1, depth])

            if self.is_train:
                if self.convolutional:
                    mean, variance = tf.nn.moments(x, [0, 1, 2])
                else:
                    mean, variance = tf.nn.moments(x, [0])

                assign_mean = self.mean.assign(mean)
                assign_variance = self.variance.assign(variance)
                with tf.control_dependencies([assign_mean, assign_variance]):
                    normed = tf.nn.batch_norm_with_global_normalization(
                        x_unflattened, mean, variance, self.beta, self.gamma, self.epsilon,
                        scale_after_normalization=self.scale_after_normalization)
            else:
                mean = self.ema.average(self.mean)
                variance = self.ema.average(self.variance)
                local_beta = tf.identity(self.beta)
                local_gamma = tf.identity(self.gamma)
                normed = tf.nn.batch_norm_with_global_normalization(
                      x_unflattened, mean, variance, local_beta, local_gamma,
                      self.epsilon, self.scale_after_normalization)
            if self.convolutional:
                return normed
            else:
                return tf.reshape(normed, [-1, depth])


def binary_cross_entropy_with_logits(logits, targets, name=None):
    """Computes binary cross entropy given `logits`.
    Here `logits` can be considered the GT and `targets` the predictions.

    For brevity, let `x = logits`, `z = targets`.  The logistic loss is

        loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))

    Args:
        logits: A `Tensor` of type `float32` or `float64`.
        targets: A `Tensor` of the same type and shape as `logits`.
        name: optional scope name
    """
    # TODO: how about using tf.nn.sigmoid_cross_entropy_with_logits here?
    # NB: when using log you always put a threshold
    eps = 1e-12
    with tf.name_scope(name, "bce_loss", [logits, targets]):
        logits = ops.convert_to_tensor(logits, name="logits")
        targets = ops.convert_to_tensor(targets, name="targets")
        return tf.reduce_mean(-(logits * tf.log(targets + eps) +
                              (1. - logits) * tf.log(1. - targets + eps)))


def conv2d(input_, output_dim,
           k_h=3, k_w=3, d_h=2, d_w=2, stddev=0.01, padding='SAME',
           use_spectral_norm=False, name="conv2d"):
    with tf.variable_scope(name):
        in_channels = input_.get_shape()[-1]
        out_channels = output_dim
        w = tf.get_variable('w', [k_h, k_w, in_channels, out_channels],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))

        if use_spectral_norm:
            w_bar = spectral_normed_weight(w, update_collection=SPECTRAL_NORM_UPDATE_OPS)
            w = w_bar

        b = tf.get_variable('b', [out_channels],
                            initializer=tf.constant_initializer(0.01))

        # if not tf.get_variable_scope().reuse:
        #     tf.summary.histogram(w.name, w)
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)
        conv = tf.nn.bias_add(conv, b)

        return conv


def deconv2d(input_, output_shape,
             k_h=3, k_w=3, d_h=2, d_w=2, stddev=0.02, padding='SAME',
             use_spectral_norm=False, name="deconv2d"):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        # TODO: 2nd param should be k_w?
        w = tf.get_variable('w', [k_h, k_h, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev, seed=4285))

        if use_spectral_norm:
            w_bar = spectral_normed_weight(w, update_collection=SPECTRAL_NORM_UPDATE_OPS)
            w = w_bar

        # if not tf.get_variable_scope().reuse:
        #     tf.summary.histogram(w.name, w)
        return tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                      strides=[1, d_h, d_w, 1], padding=padding)


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def linear(input_, output_size, stddev=0.02, use_spectral_norm=False, name='Linear'):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(name):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                initializer=tf.random_normal_initializer(stddev=stddev, seed=4285))

        variable_summaries(matrix, 'weights')

        b = tf.get_variable('b', [output_size],
                                initializer=tf.constant_initializer(0.02))

        variable_summaries(b, 'biases')

        # if not tf.get_variable_scope().reuse:
        #     tf.histogram_summary(matrix.name, matrix)
        if use_spectral_norm:
          mul = tf.matmul(input_, spectral_normed_weight(matrix, update_collection=SPECTRAL_NORM_UPDATE_OPS))
        else:
          mul = tf.matmul(input_, matrix)

        pre_act = mul + b

        variable_summaries(pre_act, 'pre_activations')

        return pre_act


def instance_norm(x):
    epsilon = 1e-9

    mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)

    return tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, epsilon))) 


def max_pool(x, filter_height, filter_width, stride_y, stride_x, name,
             padding='SAME'):
    """Create a max pooling layer."""
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)


def conv(x, num_filters, filter_height, filter_width, stride_y, stride_x, name,
         padding='SAME', groups=1):
    """Create a convolution layer.
    Adapted from: https://github.com/ethereon/caffe-tensorflow
    returns a Tensor
    """
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)

    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable('weights', shape=[filter_height,
                                                    filter_width,
                                                    input_channels/groups,
                                                    num_filters])
        biases = tf.get_variable('biases', shape=[num_filters])

    if groups == 1:
        conv = convolve(x, weights)

    # In the cases of multiple groups, split inputs & weights and
    else:
        # Split input and weights and convolve them separately
        input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
        weight_groups = tf.split(axis=3, num_or_size_splits=groups,
                                 value=weights)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

        # Concat the convolved output together again
        conv = tf.concat(axis=3, values=output_groups)

    # Add biases
    bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

    # Apply relu function
    relu = tf.nn.relu(bias, name=scope.name)

    return relu


# Taken from https://github.com/taki0112/Self-Attention-GAN-Tensorflow
def hw_flatten(x) :
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])


# Taken from https://github.com/taki0112/Self-Attention-GAN-Tensorflow
def attention(x, ch, sn=False, scope='attention', reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        f = conv2d(x, ch // 8, k_h=1, k_w=1, d_h=1, d_w=1, use_spectral_norm=sn, name='f_conv')  # [bs, h, w, c']
        g = conv2d(x, ch // 8, k_h=1, k_w=1, d_h=1, d_w=1, use_spectral_norm=sn, name='g_conv')  # [bs, h, w, c']
        h = conv2d(x, ch, k_h=1, k_w=1, d_h=1, d_w=1, use_spectral_norm=sn, name='h_conv')  # [bs, h, w, c]

        # N = h * w
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

        beta = tf.nn.softmax(s, axis=-1)  # attention map

        o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

        o = tf.reshape(o, shape=x.shape)  # [bs, h, w, C]
        x = gamma * o + x

    return x


# source: https://jhui.github.io/2017/03/12/TensorBoard-visualize-your-learning/
def variable_summaries(var, scope=None):
    pass
    # with tf.name_scope(scope):
    #     """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    #     with tf.name_scope('summaries'):
    #       mean = tf.reduce_mean(var)
    #       tf.summary.scalar('mean', mean)
    #       with tf.name_scope('stddev'):
    #         stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    #       tf.summary.scalar('stddev', stddev)
    #       tf.summary.scalar('max', tf.reduce_max(var))
    #       tf.summary.scalar('min', tf.reduce_min(var))
    #       tf.summary.histogram('histogram', var)