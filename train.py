import tensorflow as tf
from nets.w2vnet import CBOWNetwork
from utils.data_utils import DataManager
import os
# parameters
# =====================================================================
# 模型训练数据参数
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('data_path', './data/train.cbow.data', 'data_path')
tf.flags.DEFINE_string('dict_path', './data/dictionary.json', 'dict_path')

# =====================================================================
# 网络结构参数
tf.flags.DEFINE_string('network_name', 'w2w', 'network_name')
tf.flags.DEFINE_integer('embedding_size', 128, 'embedding_size')

# =====================================================================
# Word2Vec 参数
tf.flags.DEFINE_integer('window_size', 4, 'window_size')
tf.flags.DEFINE_bool('CBOW_mean', True, 'CBOW mean or sum')
tf.flags.DEFINE_string('structure', 'cbow', 'word2vec structure: cbow or skip-gram')


# =====================================================================
# 训练的参数
tf.flags.DEFINE_integer('num_sampled', 100, 'num of sampled')
tf.flags.DEFINE_integer('batch_size', 1500, 'batch_size')
tf.flags.DEFINE_integer('max_epoch', 100, 'max_epoch')
tf.flags.DEFINE_string('optimizer_name', 'adam', 'optimizer_name: adam, adadelta, adagrad')
tf.flags.DEFINE_float('learning_rate', 0.001, 'learning_rate')
tf.flags.DEFINE_float('regularization', 0.0001, 'regularization rate')

# =====================================================================
# 模型持久化 参数
tf.flags.DEFINE_string('checkpoint_dir', './running/model', 'checkpoint_dir')
tf.flags.DEFINE_integer('checkpoint_per_batch', 100, 'checkpoint_per_batch')

# =====================================================================
# 模型可视化 参数
tf.flags.DEFINE_string('summary_dir', './running/graph', 'summary_dir')



def main(_):
    # 0、模型参数校验
    if not os.path.exists(FLAGS.data_path):
        raise Exception('No data from  data_path!')
    if not os.path.exists(FLAGS.dict_path):
        raise Exception('No data from  dict_path!')
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.summary_dir):
        os.makedirs(FLAGS.summary_dir)

    assert FLAGS.structure in ['cbow', 'skip-gram'], 'only support cbow and skip-gram, check structure'
    # 一、 训练数据加载
    tf.logging.info('开始加载训练数据...')
    train_data_manager = DataManager(data_path=FLAGS.data_path,
                                     dict_path=FLAGS.dict_path,
                                     structure=FLAGS.structure,
                                     batch_size=FLAGS.batch_size,
                                     window=FLAGS.window_size,
                                     encoding='utf-8',
                                     shuffle=True)
    with tf.Graph().as_default():
        # 1. 网络构建
        model = CBOWNetwork(vocab_size=train_data_manager.word_size,
                            window=FLAGS.window_size,
                            embedding_size=FLAGS.embedding_size,
                            num_sampled=FLAGS.num_sampled,
                            is_mean=FLAGS.CBOW_mean,
                            name=FLAGS.network_name,
                            regularization=FLAGS.regularization,
                            optimizer_name=FLAGS.optimizer_name,
                            learning_rate=FLAGS.learning_rate,
                            checkpoint_dir=FLAGS.checkpoint_dir)
        # 1.1 前向网络构建
        tf.logging.info('开始构建前向网络结构...')
        model.interface()
        # 1.2 损失函数获取
        tf.logging.info('开始构建损失函数...')
        loss = model.losses()
        # 1.3 基于损失函数构建优化器和训练对象
        tf.logging.info('开始构建优化器和训练对象...')
        _, train_op = model.optimizer(loss=loss)
        # 1.4 评估值的构建
        tf.logging.info('开始进行评估...')
        model.metrics(loss=loss)
        # 1.5 构建可视化相关信息
        summary_op = tf.summary.merge_all()
        writer = tf.summary.FileWriter(FLAGS.summary_dir, graph=tf.get_default_graph())

        # 2. 模型训练
        with tf.Session() as sess:
            # 2.1 参数初始化
            tf.logging.info('开始加载模型初始化参数...')
            model.restore(session=sess)

            # 2.3 迭代训练
            for epoch in range(FLAGS.max_epoch):
                for batch_x, batch_y in train_data_manager:
                    feed_dict = {model.input_x: batch_x, model.target: batch_y}
                    _, _loss, _step, _summary = sess.run([train_op, loss, model.global_step, summary_op], feed_dict=feed_dict)
                    print('Epoch: {} step:{}, loss:{}'.format(epoch+1, _step, _loss))
                    writer.add_summary(_summary, global_step=_step)

                    if _step % FLAGS.checkpoint_per_batch == 0:
                        model.save(sess)
            model.save(sess)
            writer.close()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()