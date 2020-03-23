import tensorflow as tf
import os, argparse
from model import BiLSTM_CRF
from utils import read_corpus, read_dictionary, random_embedding, vocab_build, read_trains

# 不使用GPU，使用GPU则改为True
if False:
    # Session configuration
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
    config = tf.ConfigProto()

    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.3  # need ~700MB GPU memory
config = None
# hyper parameters
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
parser.add_argument('--mode', type=str, default='train', help='train/test/demo')
parser.add_argument('--model_path', type=str, default='correctModel', help='model for test and demo')
args = parser.parse_args()

# vocabulary build
if args.mode == "train":
    vocab_build(os.path.join('data_path', args.dataset_name, 'word2id.pkl'),
                os.path.join('data_path', args.dataset_name, 'train_data.txt'))

# get word dictionary
word2id = read_dictionary(os.path.join('data_path', args.dataset_name, 'word2id.pkl'))

# build char embeddings
embeddings = random_embedding(word2id, args.embedding_dim)  # vocab_size *

# True is 1, False is 0
tag2label = {"T":1, "F":0}


# -----------------------  path_set  ------------------------------------
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

# ---------------------------------  train or test or demo  ---------------------------
# training model  embedding shape : [voc_size, embed_dim]
if args.mode == 'train':
    train_path = os.path.join('data_path', args.dataset_name, 'train_data.txt')
    train_data, test_data = read_corpus(train_path)
    test_size = len(test_data)
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    print("train data: {}".format(len(train_data)))
    print("test data: {}".format(test_size))
    model.train(train=train_data, dev=test_data)

# testing model
elif args.mode == 'test':
    model_file = tf.train.latest_checkpoint(model_path)
    print(model_file)
    paths['model_path'] = model_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    test = read_trains(os.path.join('data_path', args.dataset_name, 'test.txt'))
    print("test data: {}".format(test))
    model.test(test)

# demo
elif args.mode == 'demo':
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        print('============= demo =============')
        saver.restore(sess, ckpt_file)
        # while 1:
        while True:
            print('Please input your sentence:')
            try:
                demo_sent = input()
            except:
                print("input error\n\n")
            if demo_sent == "quit":
                break
            elif demo_sent == '' or demo_sent.isspace():
                demo_sent = "未见明确神经脉管机。"
            demo_sent = list(demo_sent.strip())
            demo_data = [(demo_sent, ['T'] * len(demo_sent))]
            tag = model.demo_one(sess, demo_data)
            print(demo_sent, "\n", tag)
