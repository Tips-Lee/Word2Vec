import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.manifold import TSNE
import json

mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('checkpoint_dir', './running/model', 'checkpoint_dir')
tf.flags.DEFINE_string('dict_path', './data/dictionary.json', 'dict_path')


def main(_):
    checkpoint_dir = FLAGS.checkpoint_dir
    dict_path = FLAGS.dict_path
    words = json.load(open(dict_path, 'r', encoding='utf-8'))

    with tf.Graph().as_default():
        with tf.Session() as sess:
            # 1、恢复模型图和模型参数
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path and os.path.exists('{}.meta'.format(ckpt.model_checkpoint_path)):
                tf.logging.info('Restore Graph...')
                saver = tf.train.import_meta_graph('{}.meta'.format(ckpt.model_checkpoint_path))
                # saver = tf.train.Saver()

                tf.logging.info('Restore Model...')
                saver.restore(sess, ckpt.model_checkpoint_path)
                saver.recover_last_checkpoints(ckpt.all_model_checkpoint_paths[-1:])
                # saver.restore(sess, checkpoint_dir)

                tf.logging.info('Restore Completed...')
                # saver.recover_last_checkpoints()
            else:
                raise Exception('No model_checkpoint_path')

            embedding_table = tf.get_default_graph().get_tensor_by_name('w2w/embedding/embedding_table:0')
            embedding_table_ = sess.run(embedding_table)
            # print(embedding_table_.shape)

    tf.logging.info('Start Plotting...')
    tsne = TSNE(n_components=2)
    tsne_embedding_table_ = tsne.fit_transform(embedding_table_)[:1000]
    plt.figure(figsize=(10, 10))
    # plt.subplots()
    for idx in range(len(tsne_embedding_table_)):
        plt.scatter(*tsne_embedding_table_[idx], color='b')
        plt.annotate('{}_{}'.format(idx, words[idx]), [*tsne_embedding_table_[idx]], color='r', alpha=0.8)
    plt.show()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
