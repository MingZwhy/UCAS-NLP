#自然语言处理 2023 第二次作业
#使用 FNN, RNN, LSTM模型计算文本词向量
#作者：陈远腾 2020k8009929041
import sys

import data_processing
import model
import output

import argparse
import os

#程序超参数说明：

#1：读取文本路径 load_path default = "../data/English_text.txt"
#2：输出csv文件的根目录 save_csv_path default = "../result/" 默认为根目录下result文件夹
#3：词汇表长度 Num_keys default = 1000
#4：选取词向量的长度 vector_len default = 50
#5：预测序列的长度 input_len default = 5
#6：选取模型种类 model ("FNN" / "RNN" / "LSTM" / "ALL") default = "ALL"即三个模型各跑一次
#7：model训练的轮数epochs default = 15
#8：model训练的batch_size default = 128
#9：model中的Dense层/RNN层/LSTM层的核数，分别对应三个模型 default = 256
#10：validation_split, 测试集所占比例, default = 0.2

# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    # Parse command line arguments
    desc = "choose path of loading raw text and model" \
           "as for other params like paths, please change them in py directly"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "-p", "--load_text_path", required=False, default="../data/English_text.txt",
        help="if u want to load other text, please --load_text_path <your path>"
    )
    parser.add_argument(
        "-s", "--save_csv_path", required=False, default="../result/",
        help="if u want to save output to other position, --save_csv_path <your dir path>"
    )
    parser.add_argument(
        "-n", "--Num_keys", required=False, default=1000,
        help="the length of words 词汇表长度默认为1000"
    )
    parser.add_argument(
        "-v", "--vector_len", required=False, default=50,
        help="the length of word vector, default=50"
    )
    parser.add_argument(
        "-i", "--input_len", required=False, default=5,
        help="the length of predict len, default=5"
    )
    parser.add_argument(
        "-m", "--model", required=False, default="ALL",
        help="choose model, FNN / RNN / LSTM / ALL"
    )
    parser.add_argument(
        "-e", "--train_epochs", required=False, default=15,
        help="choose the num of epochs for train model"
    )
    parser.add_argument(
        "-b", "--batch_size", required=False, default=128,
        help="batch_size in training model, default = 128"
    )
    parser.add_argument(
        "-k", "--Num_kernels", required=False, default=256,
        help="num of kernels of Dense/RNN/LSTM in model, default = 256"
    )
    parser.add_argument(
        "-d", "--validation_split", required=False, default=0.2,
        help="test / (train + test), default = 0.2"
    )
    parser.add_argument(
        "-l", "--Num_likely", required=False, default=20,
        help="for any word, the number of outputing its similiar words, default = 20"
    )
    args = parser.parse_args()

    #------------------------------------------文本数据预处理-----------------------------------------
    load_text_path = args.load_text_path
    if not os.path.exists(load_text_path):
        print("path " , load_text_path , " not exists")
        sys.exit(1)

    #读取文本
    data = data_processing.read_file(load_text_path)
    #做分词、去除标点、去除停用词
    text = data_processing.split_words(data)
    text = data_processing.delete_punctuation(text)
    text = data_processing.delete_stopwords(text)

    # 得到文本出现所有词的词典，并按其出现频率由高到低排列得到列表
    all_keys_dict, all_keys = data_processing.Get_all_keys(text)

    # 从全部词的词汇表提取长度为(Num_keys+1)的词汇表
    Num_keys = args.Num_keys
    main_keys, word_index, index_word = data_processing.Get_main_keys(all_keys, Num_keys)

    #根据word_index，将原处理过的文本映射为数字列表
    num_text = data_processing.Trans_text2num_text(text, word_index)

    #根据设定的序列长度，由num_text得到dataset
    input_len = args.input_len
    dataset = data_processing.Make_dataset(num_text, input_len)

    #将dataset中的每个序列前n-1个词作为预测序列
    #最后一个词作为标签，并转换为tf_tensor
    tf_data, tf_label = data_processing.Make_data_label(dataset)

    print("finishing data processing! \n\n")

    # ------------------------------------------------------------------------------------------

    # ------------------------------------------训练model-----------------------------------------
    model_name = args.model
    vector_len = int(args.vector_len)
    train_epochs = int(args.train_epochs)
    batch_size = int(args.batch_size)
    Num_kernels = int(args.Num_kernels)
    validation_split = float(args.validation_split)

    model_fnn = model.Build_FNN_model(Num_keys, vector_len, input_len, Num_kernels)
    model_rnn = model.Build_RNN_model(Num_keys, vector_len, input_len, Num_kernels)
    model_lstm = model.Build_LSTM_model(Num_keys, vector_len, input_len, Num_kernels)

    if(model_name == "FNN"):
        print("FNN model\n")
        model_fnn.summary()

        model_fnn = model.Compile_model(model_fnn)
        model_fnn = model.Train_model(model_fnn, tf_data, tf_label, train_epochs,
                                      batch_size, validation_split)
        print("\nfinish train!\n\n")

    elif(model_name == "RNN"):
        print("RNN model\n")
        model_rnn.summary()

        model_rnn = model.Compile_model(model_rnn)
        model_rnn = model.Train_model(model_rnn, tf_data, tf_label, train_epochs,
                                      batch_size, validation_split)
        print("\nfinish train!\n\n")

    elif(model_name == "LSTM"):
        print("LSTM model\n")
        model_lstm.summary()

        model_lstm = model.Compile_model(model_lstm)
        model_lstm = model.Train_model(model_lstm, tf_data, tf_label, train_epochs,
                                      batch_size, validation_split)
        print("\nfinish train!\n\n")

    elif(model_name == "ALL"):
        print("ALL: we will train FNN, RNN, LSTM step by step")

        print("model1 : FNN model\n\n")
        model_fnn.summary()
        model_fnn = model.Compile_model(model_fnn)
        model_fnn = model.Train_model(model_fnn, tf_data, tf_label, train_epochs,
                                      batch_size, validation_split)

        print("model2 : RNN model\n\n")
        model_rnn.summary()
        model_rnn = model.Compile_model(model_rnn)
        model_rnn = model.Train_model(model_rnn, tf_data, tf_label, train_epochs,
                                      batch_size, validation_split)

        print("model3 : LSTM model\n\n")
        model_lstm.summary()

        model_lstm = model.Compile_model(model_lstm)
        model_lstm = model.Train_model(model_lstm, tf_data, tf_label, train_epochs,
                                      batch_size, validation_split)

        print("\nfinish train!\n\n")

    # ------------------------------------------------------------------------------------------

    # ------------------------------------------输出结果-----------------------------------------
    save_csv_path = args.save_csv_path
    if not os.path.exists(save_csv_path):
        os.mkdir(save_csv_path)

    Num_likely = args.Num_likely

    if(model_name == "FNN"):
        print("save the result")
        output.Output_result(model_fnn, model_name, word_index, index_word, Num_keys,
                             Num_likely, save_csv_path)

    elif(model_name == "RNN"):
        print("save the result")
        output.Output_result(model_rnn, model_name, word_index, index_word, Num_keys,
                             Num_likely, save_csv_path)

    elif(model_name == "LSTM"):
        print("save the result")
        output.Output_result(model_lstm, model_name, word_index, index_word, Num_keys,
                             Num_likely, save_csv_path)

    elif(model_name == "ALL"):
        print("save the result")
        output.Output_result(model_fnn, "FNN", word_index, index_word, Num_keys,
                             Num_likely, save_csv_path)
        output.Output_result(model_rnn, "RNN", word_index, index_word, Num_keys,
                             Num_likely, save_csv_path)
        output.Output_result(model_lstm, "LSTM", word_index, index_word, Num_keys,
                             Num_likely, save_csv_path)
    # ------------------------------------------------------------------------------------------

