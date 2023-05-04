#data_processing.py
#对输入文本进行预处理
#进行分词、去除stopwords、去除标点符号

import tensorflow as tf

import os
import sys
import string
import re
import nltk

#初次使用nltk时，需要运行
#nltk.domwload()

#读取文本文件
def read_file(file_path):
    """

    :param file_path: 文件路径
    :return: 读取的文本
    """

    try:
        with open(file_path, encoding="utf-8") as f:
            data = f.read()
    except:
        print("wrong file path!")
        sys.exit(1)

    return data

#对读取的文本进行分词
def split_words(data):
    text = nltk.word_tokenize(data)
    return text

#消除标点符号(包括中英文)
def delete_punctuation(words):
    new_words = []
    illegal_char = string.punctuation + '【·！…（）—：“”？《》、；】'
    pattern=re.compile('[%s]' % re.escape(illegal_char))
    for word in words:
        new_word = pattern.sub(u'', word)
        if not new_word == u'':
            new_words.append(new_word)
    return new_words

#消除停用词
def delete_stopwords(words):
    stops = set(nltk.corpus.stopwords.words('english'))
    new_words = [word for word in words if word.lower() not in stops]
    return new_words

#得到文本出现所有词的词典，并按其出现频率由高到低排列得到列表
def Get_all_keys(text):
    """

    :param text: 处理过的文本
    :return: 所有词的词典，词列表
    """

    fdist = nltk.probability.FreqDist(text)
    all_keys_dict = {}
    all_keys = []
    for key in fdist:
        all_keys_dict[key] = fdist[key]
        all_keys.append(key)

    return all_keys_dict, all_keys

#从全部词的词汇表提取长度为(Num_keys+1)的词汇表
#(Num_keys+1)是因为需要将词汇表之外的词映射为“UNK"
def Get_main_keys(all_keys, Num_keys):
    """

    :param all_keys: 所有词列表(由出现频率从高到低排列)
    :param Num_keys: 欲取词典长度
    :return: 前(Num_keys + 1)个词列表，其字典，及逆向字典
    """

    main_keys = all_keys[:Num_keys]

    #得到main_keys的字典 {word: num}
    word_index = dict((word, main_keys.index(word)+1) for word in main_keys)
    word_index['UNK'] = 0
    #print(word_index)

    #得到main_keys的逆向字典 {num: word}
    index_word = {}
    for k,v in word_index.items():
        index_word[v] = k

    return main_keys, word_index, index_word

#根据word_index，将原处理过的文本映射为数字列表
def Trans_text2num_text(text, word_index):
    """

    :param text: 处理过的文本
    :param word_index: {word:num}
    :return: [num1,num2,...numn]
    """

    #这里将词典里没有的词映射为0，符合我们word_index['UNK'] = 0 的设置
    f = lambda text: [word_index.get(word, 0) for word in text]
    num_text = f(text)

    return num_text

#根据设定的序列长度，由num_text得到dataset
#对main_keys中出现的每一个词，将其前面input_len个词区为序列
#而将该词本身作为label，制作预测序列和标签，若前不足vector_len个词
#则padding为0
def Make_dataset(num_text, input_len):
    """

    :param num_text: 映射为数字的处理后文本
    :param vector_len: 设置的预测序列长度
    :return: 得到总的数据集
    """

    dataset = []

    for (i, word) in enumerate(num_text):
        if (word != 0):
            if (i > input_len):
                seq = num_text[i - input_len: i + 1]
                dataset.append(seq)
            elif (i < input_len):
                seq = []
                left = input_len - i
                while (left > 0):
                    seq.append(0)
                    left = left - 1

                seq = seq + num_text[:i + 1]
                dataset.append(seq)
            else:
                seq = num_text[:i + 1]
                dataset.append(seq)

    return dataset

#将dataset中的每个序列前n-1个词作为预测序列
#最后一个词作为标签，并转换为tf_tensor
def Make_data_label(dataset):
    data = []
    label = []
    for vector in dataset:
        data.append(vector[:-1])
        label.append(vector[-1])

    tf_data = tf.constant(data)
    tf_label = tf.constant(label)

    return tf_data, tf_label