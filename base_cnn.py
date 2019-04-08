import tensorflow as tf


class BaseCNN(object):
    def __init__(self, vocabulary_size, sequence_length, num_class):
        self.embedding_size = 100
        self.learning_rate = 1e-3
        self.filter_sizes =[3, 4, 5]
        self.num_filters = 100
        self.l2_reg_lambda = 0

        self.x = tf.placeholder(tf.int32, [None, sequence_length], name="x")
        self.y = tf.placeholder(tf.int32, [None], name="y")
        self.is_training = tf.placeholder(tf.bool, [], name="is_training")
        self.global_step = tf.Variable(0, trainable=False)
        self.keep_prob = tf.where(self.is_training, 0.5, 1.0)

        l2_loss = tf.constant(0.0)

        with tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocabulary_size, self.embedding_size], -1.0, 1.0), name="W")
            self.x_emb = tf.nn.embedding_lookup(self.W, self.x)
            self.input_conv = tf.expand_dims(self.x_emb, -1)

        pooled_outputs = []
        for filter_size in self.filter_sizes:
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.input_conv,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="convolution")
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="RelU")

                pool = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length-filter_size+1, 1, 1],
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

