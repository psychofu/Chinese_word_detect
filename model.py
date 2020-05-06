import os
import sys
sys.path.append('/home/aistudio/external-libraries')
import time

import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.rnn import LSTMCell
import numpy as np

from utils import pad_sequences, batch_yield, conlleval, get_logger


class BiLSTM_CRF(object):
    def __init__(self, args, embeddings, tag2label, vocab, paths, config=None):
        self.batch_size = args.batch_size
        self.epoch_num = args.epoch
        self.hidden_dim = args.hidden_dim
        self.embeddings = embeddings
        self.CRF = args.CRF
        self.dropout_keep_prob = args.dropout
        self.optimizer = args.optimizer
        self.lr = args.lr
        self.clip_grad = args.clip
        self.tag2label = tag2label
        self.num_tags = len(tag2label)  # 2 T and F
        self.vocab = vocab
        self.shuffle = args.shuffle
        self.model_path = paths['model_path']
        self.summary_path = paths['summary_path']
        self.logger = get_logger(paths['log_path'])

        self.config = config

    def build_graph(self):
        self.add_placeholders()
        self.lookup_layer_op()
        self.biLSTM_layer_op()
        # 如果没有设置CRF层则使用soft，设置了就不使用softmox
        self.softmax_pred_op()
        # 设置crf之后loss层为crf，否则为softmax
        self.loss_op()
        # 设置优化器和梯度
        self.trainstep_op()
        # tf.global_variables_initializer()初始化变量
        self.init_op()

    def add_placeholders(self):
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")   # batch_size * max_seq_len
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")   # shape: batch_size * max_seq_len
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")     # batch_size
        self.dropout_pl = tf.placeholder(tf.float32, shape=[], name="dropout")  # 一个值
        self.lr_pl = tf.placeholder(tf.float32, shape=[], name="lr")        # 一个值

    def lookup_layer_op(self):
        with tf.variable_scope("words"):
            # shape : vocab_size * hide_dim   每一个字的embedding  这个是要训练的，已经初始化
            _word_embeddings = tf.Variable(initial_value=self.embeddings,
                                           dtype=tf.float32,
                                           trainable=True,
                                           name="_word_embeddings")
            word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings,       # 对照embedding将batch * sentence转换，batch * sentence_length * 300
                                                     ids=self.word_ids, # [[一句话的索引], [], []]
                                                     name="word_embeddings")
        # 为了防止或减轻过拟合，让某个神经元的激活值以一定的概率p，让其停止工作，这次训练过程中不更新权值，也不参加神经网络的计算
        # train时使用，test时dropout_pl设为1，即不dropout神经元
        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout_pl)      # TODO 0.5， 可考虑调小
        # 为了缓解梯度消失
        # self.word_embeddings = tf.layers.batch_normalization(word_embeddings)      # TODO 使用batch_normalization

    def biLSTM_layer_op(self):
        with tf.variable_scope("bi-lstm"):
            cell_fw = LSTMCell(self.hidden_dim)
            cell_bw = LSTMCell(self.hidden_dim)
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.word_embeddings,
                sequence_length=self.sequence_lengths,
                dtype=tf.float32)
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)     # batch_size * sent_size * 600
            output = tf.nn.dropout(output, self.dropout_pl)
            # output = tf.layers.batch_normalization(output)

        with tf.variable_scope("proj"):
            W = tf.get_variable(name="W",
                                shape=[2 * self.hidden_dim, self.num_tags],     # 这儿为什么乘2，导致后面计算的output也乘2,由于blstm返回的输出
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)

            b = tf.get_variable(name="b",
                                shape=[self.num_tags],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)

            s = tf.shape(output)    # return the shape of tensor  ：  batch_size * max_sentence_length * 600
            output = tf.reshape(output, [-1, 2 * self.hidden_dim])      # ? * 600
            # 相当于tensorflow2.0中的Dense层
            pred = tf.matmul(output, W) + b     # shape: ? * 2

            self.logits = tf.reshape(pred, [-1, s[1], self.num_tags])       # batch_size * max_sentence_length * 2

    def loss_op(self):
        if self.CRF:
            log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits,
                                                                        tag_indices=self.labels,
                                                                        sequence_lengths=self.sequence_lengths)
            self.loss = -tf.reduce_mean(log_likelihood)

        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                    labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)

        tf.summary.scalar("loss", self.loss)

    def softmax_pred_op(self):
        if not self.CRF:
            self.labels_softmax_ = tf.argmax(self.logits, axis=-1)
            self.labels_softmax_ = tf.cast(self.labels_softmax_, tf.int32)

    def trainstep_op(self):
        with tf.variable_scope("train_step"):
            # Optional `Variable` to increment by one after the variables have been updated.
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            if self.optimizer == 'Adam':
                optim = tf.train.AdamOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adadelta':
                optim = tf.train.AdadeltaOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adagrad':
                optim = tf.train.AdagradOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'RMSProp':
                optim = tf.train.RMSPropOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Momentum':
                optim = tf.train.MomentumOptimizer(learning_rate=self.lr_pl, momentum=0.9)
            elif self.optimizer == 'SGD':
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)
            else:
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)

            # 这里可以考虑不使用clip_grad，使用之后所有梯度值在这个范围内
            grads_and_vars = optim.compute_gradients(self.loss)
            grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars]
            self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)
            # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            #     grads_and_vars = optim.compute_gradients(self.loss)
            #     grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars]
            #     self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)

    def init_op(self):
        self.init_op = tf.global_variables_initializer()

    def add_summary(self, sess):
        """

        :param sess:
        :return:
        """
        # merge_all 可以将所有summary全部保存到磁盘，以便tensorboard显示。
        self.merged = tf.summary.merge_all()
        # 指定一个文件用来保存图。
        self.file_writer = tf.summary.FileWriter(self.summary_path, sess.graph)

    def train(self, train, dev):        # 这里的train和dev为所有的数据
        """

        :param train:
        :param dev:
        :return:
        """
      # In addition to checkpoint files, savers keep a protocol buffer on disk with
      # the list of recent checkpoints. This is used to manage numbered checkpoint
      # files and by `latest_checkpoint()`, which makes it easy to discover the path
      # to the most recent checkpoint. That protocol buffer is stored in a file named
      # 'checkpoint' next to the checkpoint files.
      #
      # If you create several savers, you can specify a different filename for the
      # protocol buffer file in the call to `save()`.
        saver = tf.train.Saver(tf.global_variables())   # Create a saver.

        with tf.Session(config=self.config) as sess:
            sess.run(self.init_op)
            self.add_summary(sess)

            for epoch in range(self.epoch_num):     # 迭代次数
                self.run_one_epoch(sess, train, dev, epoch, saver)

    def test(self, test):
        saver = tf.train.Saver()
        with tf.Session(config=self.config) as sess:
            self.logger.info('=========== testing ===========')
            saver.restore(sess, self.model_path)
            label_list, seq_len_list = self.dev_one_epoch(sess, test)
            leng = len(label_list)
            f = open("E:\\py_project\\Chinese_word_detect\\detectError.txt", mode="w", encoding="utf-8")
            totalwords = 0
            wrongwords = 0
            totalsent = 0
            wrongsent = 0
            for i in range(leng):
                flag = False
                totalwords = totalwords + len(label_list[i])
                totalsent = totalsent + 1
                for j in range(len(label_list[i])):
                    if label_list[i][j] == 1:
                        label_list[i][j] = "T"
                    else:
                        label_list[i][j] = "F"
                    if label_list[i][j] != test[i][1][j]:
                        print(str(i) + "  " + str(j) + ": " + test[i][0][j] + "--" + test[i][1][j])
                        wrongwords = wrongwords + 1
                        flag = True

                if flag:
                    wrongsent = wrongsent + 1

                f.write(str(label_list[i]) + "\n")
                f.write(str(test[i][1]) + "\n")
                f.write(str(test[i][0]))
                f.write("\n\n")
            f.close()
            print(wrongwords / totalwords)
            print(wrongsent / totalsent)

    def demo_one(self, sess, sent):
        """

        :param sess:
        :param sent: 
        :return:
        """
        label_list = []
        for seqs, labels in batch_yield(sent, self.batch_size, self.vocab, self.tag2label, shuffle=False):
            label_list_, _ = self.predict_one_batch(sess, seqs)
            label_list.extend(label_list_)
        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag  # if label != 0 else label
        tag = [label2tag[label] for label in label_list[0]]
        return tag

    def run_one_epoch(self, sess, train, dev, epoch, saver):
        """

        :param sess:
        :param train:
        :param dev:
        :param tag2label:
        :param epoch:
        :param saver:
        :return:
        """
        num_batches = (len(train) + self.batch_size - 1) // self.batch_size # batch_size计算需要多少个batch

        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        # 句子和tag转变为数值和01
        batches = batch_yield(train, self.batch_size, self.vocab, self.tag2label, shuffle=self.shuffle)
        for step, (seqs, labels) in enumerate(batches):     # batch_size个数据(batch_size句话)
            sys.stdout.write(' processing: {} batch / {} batches.'.format(step + 1, num_batches) + '\r')
            step_num = epoch * num_batches + step + 1
            # 这里有pad操作
            feed_dict, _ = self.get_feed_dict(seqs, labels, self.lr, self.dropout_keep_prob)
            _, loss_train, summary, step_num_ = sess.run([self.train_op, self.loss, self.merged, self.global_step],
                                                         feed_dict=feed_dict)
            if step + 1 == 1 or (step + 1) % 20 == 0 or step + 1 == num_batches:
                print('logger info')
                self.logger.info(
                    '{} epoch {}, step {}, loss: {:.4}, global_step: {}'.format(start_time, epoch + 1, step + 1,
                                                                                loss_train, step_num))

            self.file_writer.add_summary(summary, step_num)

            if step + 1 == num_batches:
                # TODO 保存文件到checkpoints
                print("----------save once--------------")
                saver.save(sess, self.model_path, global_step=step_num)

        self.logger.info('===========validation / test===========')
        label_list_dev, seq_len_list_dev = self.dev_one_epoch(sess, dev)
        self.evaluate(label_list_dev, dev)

    def get_feed_dict(self, seqs, labels=None, lr=None, dropout=None):
        """

        :param seqs:
        :param labels:
        :param lr:
        :param dropout:
        :return: feed_dict
        """
        word_ids, seq_len_list = pad_sequences(seqs, pad_mark=0)

        feed_dict = {self.word_ids: word_ids,   # shape: batch_size * max_len_snet
                     self.sequence_lengths: seq_len_list}   # shape: batch_size
        if labels is not None:
            labels_, _ = pad_sequences(labels, pad_mark=0)
            feed_dict[self.labels] = labels_
        if lr is not None:
            feed_dict[self.lr_pl] = lr
        if dropout is not None:
            feed_dict[self.dropout_pl] = dropout

        return feed_dict, seq_len_list

    def dev_one_epoch(self, sess, dev):
        """

        :param sess:
        :param dev:
        :return:
        """
        label_list, seq_len_list = [], []
        for seqs, labels in batch_yield(dev, self.batch_size, self.vocab, self.tag2label, shuffle=False):
            label_list_, seq_len_list_ = self.predict_one_batch(sess, seqs)
            label_list.extend(label_list_)
            seq_len_list.extend(seq_len_list_)
        return label_list, seq_len_list

    def predict_one_batch(self, sess, seqs):
        """

        :param sess:
        :param seqs:
        :return: label_list
                 seq_len_list
        """
        feed_dict, seq_len_list = self.get_feed_dict(seqs, dropout=1.0)     # predict时不使用dropout，故为1.0

        if self.CRF:
            logits, transition_params = sess.run([self.logits, self.transition_params],
                                                 feed_dict=feed_dict)
            label_list = []
            for logit, seq_len in zip(logits, seq_len_list):
                viterbi_seq, _ = self.viterbiDecode(logit[:seq_len], transition_params)
                label_list.append(viterbi_seq)
            return label_list, seq_len_list

        else:
            label_list = sess.run(self.labels_softmax_, feed_dict=feed_dict)
            return label_list, seq_len_list

    def viterbiDecode(self, score, transition_params):
        """Decode the highest scoring sequence of tags outside of TensorFlow.
        This should only be used at test time.
        Args:
          score: A [seq_len, num_tags] matrix of unary potentials.
          transition_params: A [num_tags, num_tags] matrix of binary potentials.
        Returns:
          viterbi: A [seq_len] list of integers containing the highest scoring tag
              indices.
          viterbi_score: A float containing the score for the Viterbi sequence.
        """
        trellis = np.zeros_like(score)
        backpointers = np.zeros_like(score, dtype=np.int32)
        trellis[0] = score[0]

        for t in range(1, score.shape[0]):
            v = np.expand_dims(trellis[t - 1], 1) + transition_params
            trellis[t] = score[t] + np.max(v, 0)
            backpointers[t] = np.argmax(v, 0)

        viterbi = [np.argmax(trellis[-1])]
        for bp in reversed(backpointers[1:]):
            viterbi.append(bp[viterbi[-1]])
        viterbi.reverse()

        viterbi_score = np.max(trellis[-1])
        return viterbi, viterbi_score

    def evaluate(self, label_list, data):
        """

        :param label_list:
        :param seq_len_list:
        :param data:
        :param epoch:
        :return:
        """
        # TODO need to see
        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag#  if label != 0 else label

        model_predict = []
        for label_, (sent, tag) in zip(label_list, data):
            tag_ = [label2tag[label__] for label__ in label_]
            sent_res = []
            if len(label_) != len(sent):
                print(sent)
                print(len(label_))
                print(tag)
            for i in range(len(sent)):
                sent_res.append([sent[i], tag[i], tag_[i]])
            model_predict.append(sent_res)
        print(label_list[0])
        self.logger.info(conlleval(model_predict))
