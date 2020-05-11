import tensorflow as tf
import os, abc


class Network(object, metaclass=abc.ABCMeta):
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

    @abc.abstractmethod
    def interface(self):
        pass

    @abc.abstractmethod
    def losses(self):
        pass

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