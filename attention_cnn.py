import tensorflow as tf


class AttentionCNN(object):
    def __init__(self, vocabulary_size, document_max_len, num_class):
        self.embedding_size = 128
        self.learning_rate = 1e-3
        self.filter_sizes = [3, 4, 5]
        self.num_filters = 100
        self.attention_dim = 100
        self.use_attention = True

        self.x = tf.placeholder(tf.int32, [None, document_max_len], name="x")
        self.y = tf.placeholder(tf.int32, [None], name="y")
        self.is_training = tf.placeholder(tf.bool, [], name="is_training")
        self.global_step = tf.Variable(0, trainable=False)
        self.keep_prob = tf.where(self.is_training, 0.5, 1.0)

        if use_attention:

            self.attention_hidden_dim = attention_dim
            self.attention_W = tf.Variable(
                tf.random_uniform([self.embedding_size, self.attention_hidden_dim], 0.0, 1.0),
                name="attention_W")
            self.attention_U = tf.Variable(
                tf.random_uniform([self.embedding_size, self.attention_hidden_dim], 0.0, 1.0),
                name="attention_U")
            self.attention_V = tf.Variable(tf.random_uniform([self.attention_hidden_dim, 1], 0.0, 1.0),
                                           name="attention_V")
            self.output_att = list()
            with tf.name_scope("attention"):
                input_att = tf.split(self.x, self.document_max_len, axis=1)
                for index, x_i in enumerate(input_att):
                    x_i = tf.reshape(x_i, [-1, self.embedding_size])
                    c_i = self.attention(x_i, input_att, index)
                    inp = tf.concat([x_i, c_i], axis=1)
                    self.output_att.append(inp)

                input_conv = tf.reshape(tf.concat(self.output_att, axis=1),
                                        [-1, self.document_max_len, self.embedding_size*2],
                                        name="input_convolution")
            self.input_x = input_conv
        else:
            self.input_x = self.x

        with tf.name_scope("embedding"):
            init_embeddings = tf.random_uniform([vocabulary_size, self.embedding_size])
            self.embeddings = tf.get_variable("embeddings", initializer=init_embeddings)
            self.x_emb = tf.nn.embedding_lookup(self.embeddings, self.input_x)
            self.x_emb = tf.expand_dims(self.x_emb, -1)

        pooled_outputs = []
        for filter_size in self.filter_sizes:
            conv = tf.layers.conv2d(
                self.x_emb,
                filters=self.num_filters,
                kernel_size=[filter_size, self.embedding_size],
                strides=(1, 1),
                padding="VALID",
                activation=tf.nn.relu)
            pool = tf.layers.max_pooling2d(
                conv,
                pool_size=[document_max_len - filter_size + 1, 1],
                strides=(1, 1),
                padding="VALID")
            pooled_outputs.append(pool)

        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, self.num_filters * len(self.filter_sizes)])

        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(h_pool_flat, self.keep_prob)

        with tf.name_scope("output"):
            self.logits = tf.layers.dense(h_drop, num_class, activation=None)
            self.predictions = tf.argmax(self.logits, -1, output_type=tf.int32)

        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def attention(self, x_i, x, index):
        e_i = []
        c_i = []

        for output in x:
            output = tf.reshape(output, [-1, self.embedding_size])
            atten_hidden = tf.tanh(tf.add(tf.matmul(x_i, self.attention_W), tf.matmul(output, self.attention_U)))
            e_i_j = tf.matmul(atten_hidden, self.attention_V)
            e_i.append(e_i_j)
        e_i = tf.concat(e_i, axis=1)
        alpha_i = tf.nn.softmax(e_i)
        alpha_i = tf.split(alpha_i, self.document_max_len, 1)
        for j, (alpha_i_j, output) in enumerate(zip(alpha_i, x)):
            if j == index:
                continue
            else:
                output = tf.reshape(output, [-1, self.embedding_size])
                c_i_j = tf.multiply(alpha_i_j, output)
                c_i.append(c_i_j)
        c_i = tf.reshape(tf.concat(c_i, axis=1), [-1, self.document_max_len-1, self.embedding_size])
        c_i = tf.reduce_sum(c_i, 1)
        return c_i
