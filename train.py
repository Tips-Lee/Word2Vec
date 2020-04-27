import tensorflow as tf


def main(_):
    with tf.Graph().as_default():
        # 1. 网络构建
        # 2. 模型训练
        with tf.Session() as sess:
            # 2.1 参数初始化
            # 2.2 训练数据加载
            # 2.3 迭代训练
            pass


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()