import tensorflow as tf
from nets.base import Network


class SkipGramNetwork(Network):
    def __init__(self, vocab_size=3000,
                 window=4,
                 embedding_size=128,
                 num_sampled=100,
                 name='skip-gram',
                 regularization=0.001,
                 optimizer_name='adam',
                 learning_rate=0.001,
                 checkpoint_dir='./running/model'):
        super(SkipGramNetwork, self).__init__(
            vocab_size=vocab_size,
            window=window,
            embedding_size=embedding_size,
            num_sampled=num_sampled,
            name=name,
            regularization=regularization,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            checkpoint_dir=checkpoint_dir)

    def interface(self):
        with tf.variable_scope(self.name):
            # 1、定义输入
            with tf.variable_scope('placeholder'):
                self.input_x = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='input_x')
                self.target = tf.placeholder(dtype=tf.int32, shape=[None, self.window], name='target')
                self.training = tf.placeholder_with_default(True, [], name='training')
                self.global_step = tf.train.get_or_create_global_step()
            # 2、 Embedding 操作将单词id转化为词向量
            with tf.variable_scope('embedding'), tf.device('/cpu:0'):
                # 2.1、定义词向量转换表
                self.embedding_table = tf.get_variable('embedding_table', [self.vocab_size, self.embedding_size], tf.float32)
                # 2.2、单词转换为词向量 [N, 1] -- [N, 1, E]
                vectors = tf.nn.embedding_lookup(self.embedding_table, self.input_x)
            # 3、 对输入值进行合并，得到最终的特征属性
            with tf.variable_scope('merge'):
                # 3.1、特征压缩 [N, 1, E] -- [N, E]
                features = tf.squeeze(vectors, axis=1)
            self.features = tf.identity(features, 'features')

    def losses(self):
        with tf.variable_scope('loss'):
            weights = tf.get_variable(name='weight', shape=[self.vocab_size, self.embedding_size])
            biases = tf.get_variable(name='bias', shape=[self.vocab_size])

            def train_loss():
                _loss = tf.nn.sampled_softmax_loss(weights=weights,
                                                   biases=biases,
                                                   labels=self.target,  # [N, num_true]
                                                   inputs=self.features,
                                                   num_sampled=self.num_sampled,
                                                   num_classes=self.vocab_size,
                                                   num_true=self.window)
                _loss = tf.reduce_mean(_loss, name='train_loss')
                return _loss

            def eval_loss():
                logits = tf.nn.bias_add(tf.matmul(self.features, weights, transpose_b=True), biases)
                labels = tf.one_hot(self.target, depth=self.vocab_size)  # [N, T] --> [N, T, V]
                labels = tf.reduce_sum(labels, axis=1)  # [N, T, V] --> [N, V]
                _loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
                _loss = tf.reduce_mean(_loss, name='eval_loss')
                return _loss
            loss = tf.cond(self.training, true_fn=train_loss, false_fn=eval_loss)
            tf.summary.scalar('loss', loss)

            l2_loss = tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases) + tf.nn.l2_loss(self.embedding_table)
            l2_loss = self.regularization * l2_loss
            tf.summary.scalar('l2_loss', l2_loss)

            total_loss = loss + l2_loss
            tf.summary.scalar('total_loss', total_loss)
            return total_loss