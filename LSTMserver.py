#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: nl8590687
语音识别API的HTTP服务器程序

"""
# import sys
# sys.path.append("./")

import http.server
import tensorflow as tf
import os, argparse
from model import BiLSTM_CRF
from utils import read_corpus, read_dictionary, random_embedding, vocab_build, read_trains
import re

config = None
parser = argparse.ArgumentParser(description='BiLSTM-CRF for Error Correction in Chinese')

# 训练数据MEDICAL路径
parser.add_argument('--dataset_name', type=str, default='MEDICAL', help='choose a dataset name, --dataset_name MEDICAL')
parser.add_argument('--batch_size', type=int, default=40, help='#sample of each minibatch')
parser.add_argument('--epoch', type=int, default=10, help='#epoch of training')
parser.add_argument('--hidden_dim', type=int, default=300, help='#dim of hidden state')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
parser.add_argument('--CRF', type=bool, default=True, help='use CRF at the top layer. if False, use Softmax')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')    # clip 梯度裁剪
parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')
parser.add_argument('--embedding_dim', type=int, default=300, help='random init char embedding_dim')
parser.add_argument('--shuffle', type=bool, default=True, help='shuffle training data before each epoch')
parser.add_argument('--mode', type=str, default='test', help='train/test/demo')
parser.add_argument('--model_path', type=str, default='correctModel', help='model for test and demo')
args = parser.parse_args()

# get word dictionary
word2id = read_dictionary(os.path.join('data_path', args.dataset_name, 'word2id.pkl'))

# build char embeddings
embeddings = random_embedding(word2id, args.embedding_dim)  # vocab_size *

# True is 1, False is 0
tag2label = {"T":1, "F":0}
paths = {}
output_path = os.path.join('model_path', args.dataset_name, args.model_path)
if not os.path.exists(output_path):
    os.makedirs(output_path)
# 保存图的文件夹
summary_path = os.path.join(output_path, "summaries")
paths['summary_path'] = summary_path
if not os.path.exists(summary_path):
    os.makedirs(summary_path)
# checkpoints文件夹
model_path = os.path.join(output_path, "checkpoints/")
if not os.path.exists(model_path):
    os.makedirs(model_path)
model_prefix = os.path.join(model_path, "model")
paths['model_path'] = model_prefix
# 结果保存文件夹
result_path = os.path.join(output_path, "results")
paths['result_path'] = result_path
if not os.path.exists(result_path):
    os.makedirs(result_path)

log_path = os.path.join(result_path, "logs.txt")
paths['log_path'] = log_path

if True:
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    saver = tf.train.Saver()
    sess = tf.Session(config=config)
    print('============= demo =============')
    saver.restore(sess, ckpt_file)



class TestHTTPHandle(http.server.BaseHTTPRequestHandler):
    def setup(self):
        self.request.settimeout(10)
        print("setup")
        http.server.BaseHTTPRequestHandler.setup(self)

    def _set_response(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        # self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

    # def do_GET(self):
    #
    #     self.protocal_version = 'HTTP/1.1'
    #     self._set_response()
    #     datas = self.rfile.read(int(self.headers['content-length']))
    #     datas = datas.decode('utf-8')
    #     print(datas)
    #     r = self.correct(datas)
    #     # self._set_response()
    #
    #     print(r)
    #     r = bytes(r, encoding="utf-8")
    #     self.wfile.write(r)

    def do_POST(self):
        # 获取post提交的数据
        self.protocal_version = 'HTTP/1.1'
        self._set_response()
        datas = self.rfile.read(int(self.headers['content-length']))
        datas = datas.decode('utf-8')
        # 打印原始文本
        print(datas)

        r = self.lstmcorrect(datas)
        # self._set_response()

        # 打印flag
        print(r)
        r = bytes(r, encoding="utf-8")
        self.wfile.write(r)

    def lstmcorrect(self, text):
        demo_sent = list(text.strip())
        demo_data = [(demo_sent, ['T'] * len(demo_sent))]  # 这里的label需要传入，不过后面不会使用，故均为T无影响
        tag = model.demo_one(sess, demo_data)
        tag = " ".join(tag)
        return str(tag)

def start_server(ip, port):
    http_server = http.server.HTTPServer((ip, int(port)), TestHTTPHandle)
    print('服务器已开启')

    try:
        http_server.serve_forever()  # 设置一直监听并接收请求
    except KeyboardInterrupt:
        pass
    http_server.server_close()
    print('HTTP server closed')


if __name__ == '__main__':
    start_server('0.0.0.0', 8001)
