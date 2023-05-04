#output.py
#1：输出训练过后的模型的embedding层的weights，即词向量
#2：计算词与词之间的相似度（以词向量之间的欧式距离表示）
#3：取与每个词最近的10个词
#4：将以上结果以csv文件格式输出

import csv
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

# 下面所有函数的整合
def Output_result(model, model_kind, word_index, index_word, Num_keys, Num_likely, save_csv_path):
    embedding_weights = Get_word_vector(model)

    word_vector_dict = Get_wordvector_dict(embedding_weights, word_index)

    distance_matrix = Calcu_distance_matrix(word_vector_dict, word_index, Num_keys)

    likely_words_word = Calue_similiar_words(distance_matrix, word_index, index_word, Num_keys, Num_likely)

    Save_result(likely_words_word, word_vector_dict, model_kind, save_csv_path, Num_likely)

# 由模型得到词向量
def Get_word_vector(model):
    """

    :param model: 训练过得模型
    :return: 得到的词向量 (即模型embedding层的weights)
    """

    #embedding层一定是网络的第一层
    embedding_layer = model.get_layer(index = 0)
    #得到embedding层的输出
    embedding_weights = embedding_layer.get_weights()

    return embedding_weights

# 由词向量和词汇表得到 {词: 词向量}字典
def Get_wordvector_dict(word_vector, word_index):
    """

    :param word_vector: 词向量
    :param word_index: 词汇表
    :return: {词: 词向量}字典
    """

    word_vector_dict = {}
    for word in word_index.keys():
        index = word_index[word]
        array = word_vector[0][index]
        np_array = np.array(array)
        word_vector_dict[word] = np_array

    return word_vector_dict

# 由词向量和词汇表得到距离矩阵
def Calcu_distance_matrix(word_vector_dict, word_index, Num_keys):
    """

    :param word_vector_dict: {词: 词向量}字典
    :param word_index: 词汇表
    :param Num_keys: 词汇表长度
    :return: 距离矩阵 [Num_keys,Num_keys]
    """

    distance_matrix = np.zeros((Num_keys + 1 , Num_keys + 1))

    for i in range(Num_keys + 1):
        word1 = list(word_index.keys())[i]
        array1 = word_vector_dict[word1]
        for j in range(Num_keys + 1):
            word2 = list(word_index.keys())[j]
            array2 = word_vector_dict[word2]
            #计算欧式距离
            distance = np.linalg.norm(array1 - array2)
            distance_matrix[i][j] = distance

    print(distance_matrix)

    return distance_matrix

# 计算与每个词最近的Num_likely个词
def Calue_similiar_words(distance_matrix, word_index, index_word, Num_keys, Num_likely):
    """

    :param distance_matrix: 距离矩阵
    :param word_index: 词汇表
    :param index_word: 逆向词汇表
    :param Num_keys: 词汇表长度
    :param Num_likely: 希望得到最近单词的个数
    :return: 与每个词最近的Num_likely个词
    """

    likely_words_word = {}

    for i in range(Num_keys + 1):
        array = np.array(distance_matrix[i])
        sorted_list = list(np.argsort(array))
        #注意这里sorted_list[0]一定是它自身，故舍去
        sorted_list_sub = sorted_list[1:Num_likely+1]

        words = []
        for j in range(len(sorted_list_sub)):
            word = index_word[sorted_list_sub[j]]
            words.append(word)

        key = (list(word_index.keys()))[i]
        likely_words_word[key] = words

    return likely_words_word

# 将结果输出为csv文件
def Save_result(likely_words_word, word_vector_dict, model_kind, save_dir_path, Num_likely):
    """

    :param likely_words_word: 存储结果的矩阵
    :param word_vector_dict: 存储词向量的字典
    :param model_kind: 模型名("FNN" / "RNN" / "LSTM")
    :param save_dir_path: 存储的根目录
    :param Num_likely: 记录的最近似的词个数
    :return:
    """

    word = []

    for key in word_vector_dict.keys():
        vector = word_vector_dict[key]
        word_tuple = (key, vector, likely_words_word[key])
        word.append(word_tuple)

    header = ["单词", "词向量", "近似top" + str(Num_likely)]

    save_all_path = ""
    if(save_dir_path[-1] != '/'):
        save_all_path = save_dir_path + "/" + model_kind + ".csv"
    else:
        save_all_path = save_dir_path + model_kind + ".csv"

    with open(save_all_path, 'w', encoding='utf-8') as file_obj:
        # 1:创建writer对象
        writer = csv.writer(file_obj)
        # 2:写表头
        writer.writerow(header)
        # 3:遍历列表，将每一行的数据写入csv
        for p in word:
            writer.writerow(p)

    print("Successfully save ", save_all_path)