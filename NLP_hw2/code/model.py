#model.py定义了用于计算词向量的三个模型
#   model1: FNN model
#   model2: RNN model
#   model3: LSTM model

# including define model, compile model and training model

import tensorflow as tf
from tensorflow import keras
from keras import layers

# model FNN, RNN, LSTM

# model1: FNN model
def Build_FNN_model(Num_keys, Vector_len, Input_len, Num_kernels):
    """

    :param Num_keys: 词汇表长度
    :param Vector_len: 单个词向量的长度
    :param Input_len: 单个输入向量的长度
    :param Num_kernels: Hidden Layer Dense层的核数
    :return: model
    """

    model = keras.Sequential()
    # Enbedding层将Num_keys个词汇嵌入到Num_keys个长度为Vector_Len的向量中
    model.add(layers.Embedding(Num_keys+1, Vector_len, input_length=Input_len))
    # FNN模型需要先将Enbedding的输入结果铺平（将向量展开）
    model.add(layers.Flatten())
    """
    Enbedding_output:  (Batch_size , Input_len , Vector_len))
                                        |
                                        |  layers.Flatten()
                                        V
    Flatten_output:    (Batch_size , (Input_len x Vector_len))   
    """
    #添加隐藏层
    model.add(layers.Dense(Num_kernels))
    #输出层, 多分类激活函数使用softmax
    model.add(layers.Dense(Num_keys+1, activation='softmax'))

    return model

# model2: RNN model
def Build_RNN_model(Num_keys, Vector_len, Input_len, Num_kernels):
    """

    :param Num_keys: 词汇表长度
    :param Vector_len: 单个词向量的长度
    :param Input_len: 单个输入向量的长度
    :param Num_kernels: RNN层的核数
    :return: model
    """

    model = keras.Sequential()
    # Enbedding层将Num_keys个词汇嵌入到Num_keys个长度为Vector_Len的向量中
    model.add(layers.Embedding(Num_keys+1, Vector_len, input_length=Input_len))
    #添加RNN层
    model.add(layers.SimpleRNN(Num_kernels))
    #输出层, 使用"softmax"作为激活函数
    model.add(layers.Dense(Num_keys+1, activation='softmax'))

    return model

# model3: LSTM model
def Build_LSTM_model(Num_keys, Vector_len, Input_len,  Num_kernels):
    """

    :param Num_keys: 词汇表长度
    :param Vector_len: 单个词向量的长度
    :param Input_len: 单个输入向量的长度
    :param Num_kernels: LSTM层的核数
    :return: model
    """

    model = keras.Sequential()
    #Enbedding层将Num_keys个词汇嵌入到Num_keys个长度为Vector_Len的向量中
    model.add(layers.Embedding(Num_keys+1, Vector_len, input_length = Input_len))
    #LSTM层
    model.add(layers.LSTM(Num_kernels))
    #输出层, 使用"softmax"作为激活函数
    model.add(layers.Dense(Num_keys+1, activation='softmax'))

    return model


# compile model
# 由于三个模型使用的优化器、损失函数及评估方法相同，故定义同一个编译函数

def Compile_model(model):
    """
    优化器：adam
    损失函数：交叉混合熵
    (由于label是直接编码而非独热码，故使用"sparse")
    评估：acc

    :param model: 搭建的模型
    :return: 编译后的模型
    """

    model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['acc'])

    return model

# train model
def Train_model(model, tf_data, tf_label, epochs, batch_size, validation_split):
    """

    :param model: 编译过的模型
    :param tf_data: 训练数据(向量序列)
    :param tf_label: 标签集
    :param epochs: 训练轮数
    :param batch_size: batch_size
    :param validation_split: 训练集/测试集划分 (0.2意味20%训练集)
    :return: 训练过的模型
    """

    model.fit(tf_data, tf_label, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
    return model