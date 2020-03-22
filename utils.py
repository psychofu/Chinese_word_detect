import os
import pickle
import random
import numpy as np
import logging

def read_corpus(corpus_path):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: data
    """
    train_data = []
    test_data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()      # 方法用于读取所有行(直到结束符 EOF)并返回列表
    sent_, tag_ = [], []
    # i = 0
    for line in lines:
        # i = i + 1
        if line != '\n':        # 语料库中的句子空一行代表一句
            # print(i, line.strip())
            [char, label] = line.strip().split()
            sent_.append(char)
            tag_.append(label)
        else:
            if random.random()>0.3:
                train_data.append((sent_, tag_))
            else:
                test_data.append((sent_, tag_))
            sent_, tag_ = [], []

    return train_data, test_data

def read_trains(corpus_path):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: data
    """
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()      # 方法用于读取所有行(直到结束符 EOF)并返回列表
    sent_, tag_ = [], []
    for line in lines:
        if line != '\n':        # 语料库中的句子空一行代表一句
            [char, label] = line.strip().split()    # 按空格split
            sent_.append(char)
            tag_.append(label)
        else:
            data.append((sent_, tag_))
            sent_, tag_ = [], []

    return data

def vocab_build(vocab_path, corpus_path, min_count=1):  # min_count设置过滤的频数
    """

    :param vocab_path:
    :param corpus_path:
    :param min_count:
    :return:
    """
    data = read_trains(corpus_path) # data[0] <class 'tuple'>: (['痛', '点', '穿', '刺', '组', '织', '中', '见', '异', '常', '征', '象', '。'], ['F', 'F', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'F', 'F', 'F', 'T'])
    word2id = {}
    for sent_, tag_ in data:
        for word in sent_:
            if word.isdigit():
                word = '<NUM>'      # 数字
            elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
                word = '<ENG>'      # 英文
            if word not in word2id:
                word2id[word] = [len(word2id) + 1, 1]   # 保留id和频数
            else:
                word2id[word][1] += 1       # 频数加1 (0代表id, 1代表频数)
    # low_freq_words = []
    # for word, [word_id, word_freq] in word2id.items():
    #     if word_freq < min_count and word != '<NUM>' and word != '<ENG>':
    #         low_freq_words.append(word)
    # for word in low_freq_words:
    #     del word2id[word]
    temp = word2id
    for word, [word_id, word_freq] in temp.items():
        if word_freq < min_count and word != '<NUM>' and word != '<ENG>':
            del word2id[word]

    # 重新排序,只保留了word和id信息, 没有频数信息
    new_id = 1
    for word in word2id.keys():
        word2id[word] = new_id
        new_id += 1
    word2id['<UNK>'] = new_id
    word2id['<PAD>'] = 0

    print(len(word2id))
    with open(vocab_path, 'wb') as fw:
        pickle.dump(word2id, fw)
        # print(word2id)

def sentence2id(sent, word2id):
    """

    :param sent:
    :param word2id:
    :return:
    """
    sentence_id = []
    for word in sent:
        if word.isdigit():
            word = '<NUM>'
        elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
            word = '<ENG>'
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id


def read_dictionary(vocab_path):
    """

    :param vocab_path:
    :return:
    """
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))
    return word2id


def random_embedding(vocab, embedding_dim):
    """

    :param vocab:
    :param embedding_dim:
    :return:
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat


def pad_sequences(sequences, pad_mark=0):
    """

    :param sequences:
    :param pad_mark:
    :return:
    """
    max_len = max(map(lambda x: len(x), sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list


def batch_yield(data, batch_size, vocab, tag2label, shuffle=False):
    """

    :param data:
    :param batch_size:
    :param vocab:
    :param tag2label:
    :param shuffle:
    :return:
    """
    if shuffle:
        random.shuffle(data)

    seqs, labels = [], []
    for (sent_, tag_) in data:
        sent_ = sentence2id(sent_, vocab)
        label_ = [tag2label[tag] for tag in tag_]

        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(label_)

    if len(seqs) != 0:      # 最后的batch
        yield seqs, labels

def conlleval(label_predict):
    """

    :param label_predict:
    :param label_path:
    :param metric_path:
    :return:
    """

    # 计算precision和recall，计算F值
    TrueP_num = 0
    FalseP_num = 0
    FalseN_num = 0

    for sent_result in label_predict:
        for char, tag, tag_ in sent_result:
            if tag == "T" and tag_ == "T":
                TrueP_num = TrueP_num + 1
            elif tag == "F" and tag_ == "T":
                FalseP_num = FalseP_num + 1
            elif tag == "T" and tag_ == "F":
                FalseN_num = FalseN_num + 1

    Precision = TrueP_num / (TrueP_num + FalseP_num)    # 表征分类器的分类效果(查准效果)，它是在预测为正样本的实例中预测正确的频率值
    Recall = TrueP_num / (TrueP_num + FalseN_num)       # 表征某个类的召回(查全)效果，它是在标签为正样本的实例中预测正确的频率
    F_value = 2 * Precision * Recall / (Precision + Recall)

    info = "precision: {}, recall: {}, F_value: {}".format(Precision, Recall, F_value)

    return info

def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    if not logger.handlers:
        logger.addHandler(handler)
    return logger
# if __name__ == '__main__':
#     word2id = read_dictionary(os.path.join('data_path', 'MSRA', 'word2id.pkl'))
#     build_character_embeddings('./sgns.wiki.char', './vectors.npy', word2id, 300)
