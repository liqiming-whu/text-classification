import tensorflow as tf
import numpy as np


class AttentionCNN(object):
    def __init__(self, vocabulary_size, sequence_length, num_class, use_attention_input=True, use_attention_conv=True):
        self.embedding_size = 100
        self.learning_rate = 1e-3
        self.filter_sizes =[3, 4, 5]
        self.num_filters = 100
        self.atten_size = 50
        self.l2_reg_lambda = 0

        self.x = tf.placeholder(tf.int32, [None, sequence_length], name="x")
        self.y = tf.placeholder(tf.int32, [None], name="y")
        self.is_training = tf.placeholder(tf.bool, [], name="is_training")
        self.global_step = tf.Variable(0, trainable=False)
        self.keep_prob = tf.where(self.is_training, 0.5, 1.0)
        self.sequence_length = sequence_length

        l2_loss = tf.constant(0.0)

        with tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocabulary_size, self.embedding_size], -1.0, 1.0), name="W")
            self.x_emb = tf.nn.embedding_lookup(self.W, self.x)

        if use_attention_input:
            atten_out, alphas = self.attention(self.x_emb, self.atten_size)
            input_x = tf.reshape(atten_out, [-1, sequence_length, self.embedding_size])
            self.input_conv = tf.expand_dims(input_x, -1)
        else:
            self.input_conv = tf.expand_dims(self.x_emb, -1)

        pooled_outputs = []
        for filter_size in self.filter_sizes:
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                x = self.pad_wideconv(self.input_conv, filter_size)
                filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                conv = tf.nn.conv2d(
                    x,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="convolution")
                wide_h = tf.nn.relu(tf.nn.bias_add(conv, b), name="RelU")

                if use_attention_conv:
                    atten_input = tf.reshape(wide_h,
                    [-1, sequence_length + filter_size - 1, self.num_filters])
                    atten_out, alphas = self.attention(atten_input, self.atten_size)
                    wide_h_atten = tf.reshape(atten_out,
                    [-1, sequence_length+filter_size-1, 1, self.num_filters])

                else:
                    wide_h_atten = wide_h
                    
                h = tf.nn.avg_pool(
                    wide_h_atten,
                    ksize=[1, filter_size, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="h")

                pool = tf.nn.avg_pool(
                    h,
                    ksize=[1, sequence_length, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="pool")
                pooled_outputs.append(pool)

        num_filters_total = self.num_filters * len(self.filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(h_pool_flat, self.keep_prob)

        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_class],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_class]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(h_drop, W, b, name="logits")
            self.predictions = tf.cast(tf.argmax(self.logits, -1), tf.int32)

        with tf.name_scope("loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y)
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * l2_loss
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def attention(self, atten_inputs, atten_size):
        max_time = int(atten_inputs.shape[1])
        combined_hidden_size = int(atten_inputs.shape[2])
        W_omega = tf.Variable(tf.random_normal([combined_hidden_size, atten_size], stddev=0.1, dtype=tf.float32))
        b_omega = tf.Variable(tf.random_normal([atten_size], stddev=0.1, dtype=tf.float32))
        u_omega = tf.Variable(tf.random_normal([atten_size], stddev=0.1, dtype=tf.float32))

        v = tf.tanh(tf.matmul(tf.reshape(atten_inputs, [-1, combined_hidden_size]), W_omega) + tf.reshape(b_omega, [1, -1]))
        vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
        exps = tf.reshape(tf.exp(vu), [-1, max_time])
        alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])
        atten_outs = max_time *(
                atten_inputs * tf.reshape(alphas, [-1, max_time, 1]))
        return atten_outs, alphas

    def pad_wideconv(self, x, filter_size):
        return tf.pad(
            x, np.array([[0, 0], [filter_size - 1, filter_size - 1], [0, 0], [0, 0]]),
            "CONSTANT", name="pad_wideconv")
