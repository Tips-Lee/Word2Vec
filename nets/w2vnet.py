import tensorflow as tf
import os


class CBOWNetwork(object):
    def __init__(self, vocab_size=3000,
                 window=4,
                 embedding_size=128,
                 num_sampled=100,
                 is_mean=True,
                 name='CBOW',
                 regularization=0.001,
                 optimizer_name='adam',
                 learning_rate=0.001,
                 checkpoint_dir='./running/model'):
        self.name = name
        self.num_sampled = num_sampled
        self.window = window
        self.is_mean = is_mean
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.input_x = None  # [N, W]
        self.target = None  # [N, 1]
        self.training = None  # []
        self.global_step = None  # []
        self.features = None  # [N, E]
        self.embedding_table = None
        self.regularization = regularization
        self.optimizer_name = optimizer_name.lower()
        self.learning_rate = learning_rate
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.epsilon = 1e-8  # [adam, adadelta]
        self.rho = 0.95  # adadelta
        self.saver = None
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_path = os.path.join(self.checkpoint_dir, '{}.ckpt'.format(self.name))
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def interface(self):
        with tf.variable_scope(self.name):
            # 1、定义输入
            with tf.variable_scope('placeholder'):
                self.input_x = tf.placeholder(dtype=tf.int32, shape=[None, self.window], name='input_x')
                self.target = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='target')
                self.training = tf.placeholder_with_default(True, [], name='training')
                self.global_step = tf.train.get_or_create_global_step()
            # 2、 Embedding 操作将单词id转化为词向量
            with tf.variable_scope('embedding'), tf.device('/cpu:0'):
                # 2.1、定义词向量转换表
                self.embedding_table = tf.get_variable('embedding_table', [self.vocab_size, self.embedding_size], tf.float32)
                # 2.2、单词转换为词向量 [N, T] -- [N, T, E]
                vectors = tf.nn.embedding_lookup(self.embedding_table, self.input_x)
            # 3、 对输入值进行合并，得到最终的特征属性
            with tf.variable_scope('merge'):
                if self.is_mean:
                    features = tf.reduce_mean(vectors, axis=1, keepdims=False)
                else:
                    features = tf.reduce_sum(vectors, axis=1, keepdims=False)

            self.features = tf.identity(features, 'features')

    def losses(self):
        with tf.variable_scope('loss'):
            weights = tf.get_variable(name='weight', shape=[self.vocab_size, self.embedding_size])
            biases = tf.get_variable(name='bias', shape=[self.vocab_size])

            def train_loss():
                _loss = tf.nn.sampled_softmax_loss(weights=weights, biases=biases, labels=self.target, inputs=self.features,
                                           num_sampled=self.num_sampled, num_classes=self.vocab_size, num_true=1)
                _loss = tf.reduce_mean(_loss, name='train_loss')
                return _loss

            def eval_loss():
                logits = tf.nn.bias_add(tf.matmul(self.features, weights, transpose_b=True), biases)
                labels = tf.reshape(self.target, [-1])
                _loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
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

    def optimizer(self, loss, *args):
        with tf.variable_scope('train'):
            if self.optimizer_name == 'adam':
                opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.adam_beta1, beta2=self.adam_beta2, epsilon=self.epsilon)
            elif self.optimizer_name == 'adadelta':
                opt = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate, rho=self.rho, epsilon=self.epsilon)
            elif self.optimizer_name == 'adagrad':
                opt = tf.train.AdagradOptimizer(learning_rate=self.learning_rate, initial_accumulator_value=0.1)
            else:
                opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

            train_op = opt.minimize(loss, global_step=self.global_step)
            return opt, train_op

    def metrics(self, loss=None):
        pass

    def restore(self, session=None):
        if self.saver is None:
            self.saver = tf.train.Saver()
        session.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir=self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            tf.logging.info('Restore model from {}'.format(ckpt.model_checkpoint_path))
            self.saver.restore(session, ckpt.model_checkpoint_path)
            self.saver.recover_last_checkpoints(ckpt.all_model_checkpoint_paths)

    def save(self, session=None):
        if self.saver is None:
            self.saver = tf.train.Saver()
        tf.logging.info('Save model to {}'.format(self.checkpoint_path))
        self.saver.save(session, save_path=self.checkpoint_path, global_step=self.global_step)


class SkipGramNetwork(object):
    def __init__(self, vocab_size=3000,
                 window=4,
                 embedding_size=128,
                 num_sampled=100,
                 name='skip-gram',
                 regularization=0.001,
                 optimizer_name='adam',
                 learning_rate=0.001,
                 checkpoint_dir='./running/model'):
        self.name = name
        self.num_sampled = num_sampled
        self.window = window
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.input_x = None  # [N, 1]
        self.target = None  # [N, T]
        self.training = None  # []
        self.global_step = None  # []
        self.features = None  # [N, E]
        self.embedding_table = None  # [V, E]
        self.regularization = regularization
        self.optimizer_name = optimizer_name.lower()
        self.learning_rate = learning_rate
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.epsilon = 1e-8  # [adam, adadelta]
        self.rho = 0.95  # adadelta
        self.saver = None
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_path = os.path.join(self.checkpoint_dir, '{}.ckpt'.format(self.name))
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

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

    def optimizer(self, loss, *args):
        with tf.variable_scope('train'):
            if self.optimizer_name == 'adam':
                opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.adam_beta1, beta2=self.adam_beta2, epsilon=self.epsilon)
            elif self.optimizer_name == 'adadelta':
                opt = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate, rho=self.rho, epsilon=self.epsilon)
            elif self.optimizer_name == 'adagrad':
                opt = tf.train.AdagradOptimizer(learning_rate=self.learning_rate, initial_accumulator_value=0.1)
            else:
                opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

            train_op = opt.minimize(loss, global_step=self.global_step)
            return opt, train_op

    def metrics(self, loss=None):
        pass

    def restore(self, session=None):
        if self.saver is None:
            self.saver = tf.train.Saver()
        session.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir=self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            tf.logging.info('Restore model from {}'.format(ckpt.model_checkpoint_path))
            self.saver.restore(session, ckpt.model_checkpoint_path)
            self.saver.recover_last_checkpoints(ckpt.all_model_checkpoint_paths)

    def save(self, session=None):
        if self.saver is None:
            self.saver = tf.train.Saver()
        tf.logging.info('Save model to {}'.format(self.checkpoint_path))
        self.saver.save(session, save_path=self.checkpoint_path, global_step=self.global_step)
