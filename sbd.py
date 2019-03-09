from ops_alex import *
# import tensorflow.contrib.slim as slim



def decoder_sbd(inputs, image_size, batch_size, feature_size, scope='g_decsbd', reuse=False):
    with tf.variable_scope(scope, [inputs]) as _:
        print('decoder_sbd -->')
        print('inputs: %s' % inputs.shape)

        assert inputs.shape[0] == batch_size
        assert inputs.shape[1] == feature_size

        d = w = image_size
        z_b = tf.tile(inputs, [1, d * w])
        print("z_b.shape:", z_b.shape)
        z_b = tf.reshape(z_b, [batch_size, d, w, feature_size])
        print("z_b.shape: ", z_b.shape)

        x = tf.linspace(tf.constant(-1, tf.float32), tf.constant(1, tf.float32), w)
        y = tf.linspace(tf.constant(-1, tf.float32), tf.constant(1, tf.float32), w)

        xb, yb = tf.meshgrid(x, y)
        print("xb.shape: ", xb.shape)

        xb = tf.expand_dims(xb, 2)
        print("xb.shape: ", xb.shape)
        yb = tf.expand_dims(yb, 2)
        print("yb.shape: ", yb.shape)

        def pe(e):
            # print('shape:', e.shape)
            # print('e: ', e)
            res = tf.concat(axis=2, values=[e, xb, yb])
            print('res.shape:', res.shape)
            return res

        z_sb = tf.map_fn(lambda m: pe(m), z_b)
        print("z_sb:", z_sb)

        assert z_sb.shape[0] == batch_size
        assert z_sb.shape[1] == image_size
        assert z_sb.shape[2] == image_size
        assert z_sb.shape[3] == feature_size + 2

        # stack = slim.conv2d(inputs, n_filters_first_conv, [3, 3], scope='first_conv', activation_fn=None,
        #                  weights_initializer=tf.random_normal_initializer(stddev=0.02, seed=4285), biases_initializer=tf.constant_initializer(0.01))

        net = conv2d(z_sb, 320, k_h=4, k_w=4, d_h=1, d_w=1, use_spectral_norm=True, name='conv_1')
        # preact = instance_norm(inputs)
        net = tf.nn.relu(net)
        net = conv2d(net, 224, k_h=4, k_w=4, d_h=1, d_w=1, use_spectral_norm=True, name='conv_2')
        net = tf.nn.relu(net)
        net = conv2d(net, 128, k_h=4, k_w=4, d_h=1, d_w=1, use_spectral_norm=True, name='conv_3')
        net = tf.nn.relu(net)
        net = conv2d(net, 64, k_h=4, k_w=4, d_h=1, d_w=1, use_spectral_norm=True, name='conv_4')
        net = tf.nn.relu(net)
        net = conv2d(net, 32, k_h=4, k_w=4, d_h=1, d_w=1, use_spectral_norm=True, name='conv_5')
        net = tf.nn.relu(net)
        net = conv2d(net, 16, k_h=4, k_w=4, d_h=1, d_w=1, use_spectral_norm=True, name='conv_6')
        net = tf.nn.relu(net)
        net = conv2d(net, 3, k_h=4, k_w=4, d_h=1, d_w=1, use_spectral_norm=True, name='conv_7')
        net = tf.nn.relu(net)

        assert net.shape[0] == batch_size
        assert net.shape[1] == 64
        assert net.shape[2] == 64
        assert net.shape[3] == 3

        print('decoder_sbd <--')

        return tf.nn.tanh(net)
