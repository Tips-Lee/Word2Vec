import tensorflow as tf
from nets.w2vnet import CBOWNetwork
from utils.data_utils import DataManager


def main(_):
    with tf.Graph().as_default():
        # 1. 网络构建
        model = CBOWNetwork()
        # 1.1 前向网络构建
        tf.logging.info('开始构建前向网络结构...')
        model.interface()
        # 1.2 损失函数获取
        tf.logging.info('开始构建损失函数...')
        loss = model.losses()
        # 1.3 基于损失函数构建优化器和训练对象
        tf.logging.info('开始构建优化器和训练对象...')
        optimizer, train_op = model.optimizer(loss=loss)
        # 1.4 评估值的构建
        tf.logging.info('开始进行评估...')
        model.metrics(loss=loss)

        # 2. 模型训练
        with tf.Session() as sess:
            # 2.1 参数初始化
            tf.logging.info('开始加载模型初始化参数...')
            model.restore(session=sess)
            # 2.2 训练数据加载
            tf.logging.info('开始加载训练数据...')
            train_data_manager = DataManager(data_path='./data/train.cbow.data', dict_path='./data/dictionary.json',
                                             structure='cbow', batch_size=8, window=4, encoding='utf-8', shuffle=True)
            # 2.3 迭代训练
            for epoch in range(10):
                for batch_x, batch_y in train_data_manager:
                    feed_dict = {model.input_x: batch_x, model.target: batch_y}
                    _, _loss = sess.run([train_op, loss], feed_dict=feed_dict)
                    print(_loss)



if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()