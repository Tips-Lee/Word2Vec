import jieba
import os, json, re
from collections import Counter, defaultdict
PAD = '<PAD>'
UNKNOWN = '<UNKNOWN>'


def split_sentence(sentence):
    for word in jieba.cut(sentence):
        re_han_cut_all = re.compile("([\u4E00-\u9FD5]+)", re.U)
        if re_han_cut_all.match(word):
            yield word
        else:
            for char in word:
                yield char


def convert_sentence_to_words(infile, outfile, encoding='utf-8'):
    '''
    对输入文件进行分词处理，结果保存到输出文件
    :param infile:
    :param outfile:
    :param encoding:
    :return:
    '''
    with open(infile, 'r', encoding=encoding) as reader:
        with open(outfile, 'w', encoding=encoding) as writer:
            for sentence in reader:
                sentence = sentence.strip()
                if len(sentence)==0:
                    continue
                words = split_sentence(sentence)
                result = ' '.join(words)
                writer.writelines('%s\n' % result)


def build_dict(infile, out_file, min_count=5, encoding='utf-8'):
    words = defaultdict(int)
    with open(infile, 'r', encoding=encoding) as reader:
        for sentence in reader:
            sentence = sentence.strip()
            _words = sentence.split(' ')
            for word in _words:
                if len(word) > 0:
                    words[word] += 1
        words = sorted(map(lambda x: x[0], filter(lambda x: x[1] > min_count, words.items())))
        words = [PAD, UNKNOWN] + words
        dirname = os.path.dirname(out_file)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        json.dump(words, open(out_file, 'w', encoding=encoding), indent=0, ensure_ascii=False)
'''
def build_dict(infile, out_file, min_count=5, encoding='utf-8'):
    words = Counter()
    with open(infile, 'r', encoding=encoding) as reader:
        for sentence in reader:
            sentence = sentence.strip()
            _words = sentence.split(' ')
            words = words + Counter(_words)
        words = sorted(map(lambda x: x[0], filter(lambda x: x[1] > min_count, words.items())))
        words = [PAD, UNKNOWN] + words
        dirname = os.path.dirname(out_file)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        json.dump(words, open(out_file, 'w', encoding=encoding), indent=0, ensure_ascii=False)
'''


def build_record(words, window=4, structure='cbow', allow_padding=True):
    n_words = len(words)
    start, end = (0, n_words) if allow_padding else (window//2, n_words-window//2)
    for idx in range(start, end):

        center_word = words[idx]
        surrounding_words = words[idx-window//2:idx]
        surrounding_words = [PAD]*(window//2 - len(surrounding_words)) + surrounding_words
        surrounding_words = surrounding_words + words[idx + 1:idx+1 + window - len(surrounding_words)]
        surrounding_words = surrounding_words + [PAD] * (window - len(surrounding_words))
        if structure == 'cbow':
            yield surrounding_words + [center_word]
        else:
            yield [center_word] + surrounding_words


def convert_words_to_record(infile, outfile, encoding='utf-8', window=4, structure='cbow', allow_padding=True):
    with open(infile, 'r', encoding=encoding) as reader:
        with open(outfile, 'w', encoding=encoding) as writer:
            for sentence in reader:
                words = sentence.strip().split(' ')
                if len(words)==0:
                    continue
                if not allow_padding and len(words) <= window:
                    continue
                for record in build_record(words, window, structure, allow_padding):
                    writer.writelines('%s\n' % ' '.join(record))


if __name__ == '__main__':
    words = ['我', '是','中国人','，','我','爱','中国']
    ans = build_record(words, window=4, structure='cbow')
    for i in ans:
        print(i)