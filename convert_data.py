from utils.data_utils import convert_sentence_to_words, build_dict, convert_words_to_record
import tensorflow as tf
import os


tf.app.flags.DEFINE_string('opt', 'split', 'operation type: split: cut words, dictionary: build dict, record: build_record')
tf.app.flags.DEFINE_string('split_input_file', None, 'split_input_file path')
tf.app.flags.DEFINE_string('split_output_file', None, 'split_out_file path')
tf.app.flags.DEFINE_string('dict_input_file', None, 'dict_input_file path')
tf.app.flags.DEFINE_string('dict_output_file', None, 'dict_input_file path')
tf.app.flags.DEFINE_string('record_input_file', None, 'record_input_file path')
tf.app.flags.DEFINE_string('record_output_file', None, 'record_output_file path')
tf.app.flags.DEFINE_string('structure', 'cbow', 'data structure: cbow or skip-gram')
tf.app.flags.DEFINE_boolean('allow_padding', True, 'allow_padding: True or False')
tf.app.flags.DEFINE_integer('dict_min_count', 5, 'dict_min_count: ')
tf.app.flags.DEFINE_integer('window', 4, 'window: ')
FLAGS = tf.app.flags.FLAGS


def main(_):
    opt = FLAGS.opt.lower()
    if opt == 'split':
        tf.logging.info('word cut...')
        split_input_file = FLAGS.split_input_file
        split_output_file = FLAGS.split_output_file
        if split_input_file is None or not os.path.isfile(split_input_file):
            raise Exception('check out split_input_file')
        if split_output_file is None:
            raise Exception('check out split_output_file')
        convert_sentence_to_words(split_input_file, split_output_file)
        tf.logging.info('word cut completed !')

    elif opt == 'dictionary':
        tf.logging.info('build dict...')
        dict_input_file = FLAGS.dict_input_file
        dict_output_file = FLAGS.dict_output_file
        dict_min_count = FLAGS.dict_min_count
        if dict_input_file is None or not os.path.isfile(dict_input_file):
            raise Exception('check out split_input_file')
        if dict_output_file is None:
            raise Exception('check out split_output_file')
        build_dict(dict_input_file, dict_output_file, dict_min_count)
        tf.logging.info('build dict completed !')

    elif opt == 'record':
        tf.logging.info('build training record...')
        record_input_file = FLAGS.record_input_file
        record_output_file = FLAGS.record_output_file
        window = FLAGS.window
        allow_padding = FLAGS.allow_padding
        structure = FLAGS.structure.lower()
        if record_input_file is None or not os.path.isfile(record_input_file):
            raise Exception('check out record_input_file')
        if record_output_file is None:
            raise Exception('check out record_output_file')
        convert_words_to_record(record_input_file, record_output_file, window=window, structure=structure, allow_padding=allow_padding)
        tf.logging.info('build record completed !')
    else:
        tf.logging.WARN('parameter failure, check opt')


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()