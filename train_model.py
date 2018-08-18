import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import re
import sys
import pickle
import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from tqdm import tqdm
from keras import layers, callbacks
from keras.models import Model
from common import Route, LookupTable, Visualizer
from graph import Graph
from keras_utils import UpdateMonitor, Logger
from misc_utils import Flush
from keras.utils import plot_model

#
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

def get_session(gpu_fraction = 0.3):
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = gpu_fraction)
    if num_threads:
        return tf.Session(config = tf.ConfigProto(gpu_options = tpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options = gpu_options))

#
graph = Graph(topo_file='topo9.yaml')
NAME = 'RV4'
parser = argparse.ArgumentParser(prog=NAME)
parser.add_argument('-data', type=int, default=1, help='source of data')
parser.add_argument('-redirect', type=int, choices=(0, 1), default=0, help='redirect stdout to logfile')
args = parser.parse_args()

class LossHistory(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        #self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        #self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        #self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        #self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        #self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        #self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        #plt.plot(iters, self.accuracy[loss_type], 'r', label='train accuracy')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            #plt.plot(iters, self.val_acc[loss_type], 'b', label='validate accuracy')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='validate loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('loss')
        plt.legend(loc="right")


def _tqdm(iterable, desc=None):
    return iterable if args.redirect else tqdm(iterable, desc)


def load_data_new(data_file_path, pkl_file_path, from_pkl=False, save_pkl=False):
    if from_pkl:
        if os.path.exists(pkl_file_path):
            print('loading data from', pkl_file_path)
            with open(pkl_file_path, 'rb') as fr:
                routes, routes_test, min_node_index, max_node_index, min_node_test_index, max_node_test_index = pickle.load(fr)
                return routes, routes_test, min_node_index, max_node_index, min_node_test_index, max_node_test_index
        else:
            print('pkl file not found, load from txt file instead')
            return load_data_new(data_file_path, pkl_file_path, from_pkl=False, save_pkl=True)
    else:
        routes = []
        routes_test = []
        with open(data_file_path, 'r', encoding='gb2312') as fr:
            in_existing, in_current = False, False
            in_existing_test, in_current_test = False, False
            first_current = False
            first_current_test = False
            train_or_test = False # False means train data, True means test data
            hidden_or_not = False # False means not hidden, True means hidden
            min_node_index, max_node_index = np.inf, -np.inf
            min_node_test_index, max_node_test_index = np.inf, -np.inf
            for line in _tqdm(fr, desc='loading'):
                if 'Duplicate' in line or 'Types Above' in line:
                    continue
                line = line.strip()
                if line.startswith('---TestExisting'):
                    route_test = Route()
                    in_existing_test, in_current_test = True, False
                    first_current_test = True
                    train_or_test = True
                if line.startswith('---TestCurrent'):
                    in_existing_test, in_current_test = False, True
                    train_or_test = True
                if line.startswith('---Existing'):
                    route = Route()
                    in_existing, in_current = True, False
                    first_current = True
                    train_or_test = False
                if line.startswith('---Current'):
                    in_existing, in_current = False, True
                    train_or_test = False
                if line.startswith('Hidden'):
                    hidden_or_not = True
                if re.match(r'((-)?\d+\s+)+\d+', line):
                    if hidden_or_not:
                        hidden_route = [int(x) for x in line.split()]
                        if in_existing_test:
                            route_test.intra_capacity = hidden_route
                        elif in_existing:
                            route.intra_capacity = hidden_route
                        hidden_or_not = False
                    else:
                        if train_or_test:
                            seqsTest = [int(x) for x in line.split()]
                            if in_existing_test:
                                route_test.inter_capacity = seqsTest
                            if in_current_test and first_current_test:
                                node_seq_test_name = [x for x in seqsTest]
                                node_seq_test_index = graph.get_index_route(node_seq_test_name)
                                min_node_test_index = int(np.min((min_node_test_index, np.min(node_seq_test_index))))
                                max_node_test_index = int(np.max((max_node_test_index, np.max(node_seq_test_index))))
                                route_test.index_path = node_seq_test_index
                                routes_test.append(route_test)  # add to routes
                                first_current_test = False  # don't forget this!
                        else:
                            seqs = [int(x) for x in line.split()]
                            if in_existing:
                                route.inter_capacity = seqs
                            if in_current and first_current:
                                node_seq_name = [x for x in seqs]
                                node_seq_index = graph.get_index_route(node_seq_name)
                                min_node_index = int(np.min((min_node_index, np.min(node_seq_index))))
                                max_node_index = int(np.max((max_node_index, np.max(node_seq_index))))
                                route.index_path = node_seq_index
                                routes.append(route)  # add to routes
                                first_current = False  # don't forget this!
        if save_pkl and not os.path.exists(pkl_file_path):
            with open(pkl_file_path, 'wb') as fw:
                pickle.dump((routes, routes_test, min_node_index, max_node_index, min_node_test_index, max_node_test_index), fw)
            print('routes info saved in', pkl_file_path)
        return routes, routes_test, min_node_index, max_node_index, min_node_test_index, max_node_test_index

## =============== process ===============
def data_process(routes, routes_test, min_node_index, max_node_index, min_node_test_index, max_node_test_index):
    max_inter_capacity_size = int(np.max([len(route.inter_capacity) for route in routes]))
    max_intra_capacity_size = int(np.max([len(route.intra_capacity) for route in routes]))
    has_congestion = False
    congestion_node = 'none'
    if max_node_index == 61:  # has network congestion
        print('train set has congestion, processing')
        has_congestion = True
        congestion_node = max_node_index + 1
        min_node_index = 1

    print('training nodes and paths statistics:')
    print(' - samples:', len(routes))
    print(' - node range: {} - {}'.format(min_node_index, max_node_index))
    print(' - congestion:', congestion_node)
    print(' - inter cap length:', max_inter_capacity_size)
    print(' - hidden cap length:', max_intra_capacity_size)

    max_inter_capacity_size_test = int(np.max([len(route_test.inter_capacity) for route_test in routes_test]))
    max_intra_capacity_size_test = int(np.max([len(route_test.intra_capacity) for route_test in routes_test]))
    # max_history_test = int(np.max([len(route_test.histories) for route_test in routes_test]))
    # check congestion
    has_congestion_test = False
    congestion_node_test = 'none'
    if max_node_test_index == 61:  # original -1, has network congestion
        print('test set has congestion, processing')
        has_congestion_test = True
        congestion_node_test = max_node_test_index + 1
        min_node = 1

    print('testing nodes and paths statistics:')
    print(' - samples:', len(routes_test))
    print(' - node range: {} - {}'.format(min_node_test_index, max_node_test_index))
    print(' - congestion:', congestion_node_test)
    print(' - inter cap length:', max_inter_capacity_size_test)
    print(' - hidden cap length:', max_intra_capacity_size_test)

    return has_congestion, congestion_node, max_inter_capacity_size, has_congestion_test, congestion_node_test, max_inter_capacity_size_test, max_intra_capacity_size_test

def make_train_data(ntable, routes, max_inter_capacity_size, maxlen_que, maxlen_ans, features):
    print('creating training dataset ...')
    n_samples = len(routes)
    cap_train = np.zeros((n_samples, max_inter_capacity_size), dtype=np.int8)
    que_train = np.zeros((n_samples, maxlen_que, features), dtype=np.int8)
    ans_train = np.zeros((n_samples, maxlen_ans, features), dtype=np.int8)

    for i, route in _tqdm(enumerate(routes), desc='encoding'):
        # print('route',route.path, 'route name:', graph.get_route_name(route.path))
        cap_train[i] = route.inter_capacity
        que_train[i] = ntable.encode([route.index_path[0], route.index_path[-1]], maxlen_que)
        ans_train[i] = ntable.encode(route.index_path[1:-1], maxlen_ans)  # answer must be encoded!

    print('training data shape:')
    print(' - inter-capacity:', cap_train.shape)
    print(' - question:', que_train.shape)
    print(' - answer:', ans_train.shape)

    # shuffle data before train/test split
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    cap_train = cap_train[indices]
    que_train = que_train[indices]
    ans_train = ans_train[indices]

    return cap_train, que_train, ans_train

# data path configs, Mac version
#_DATA_DIR = os.path.join(os.path.expanduser('~/datasets/routing'), 'data0524-{}'.format(args.data))

# data path configs, server version
#_DATA_FILE_NAME = 'output_BRPC.dat'
_DATA_DIR = os.path.join(os.path.expanduser('~/Zhong_Exp/Datasets/routing'), 'data-{}'.format(args.data))
_TRAINLOGS_DIR = os.path.join(os.path.expanduser('~/Zhong_Exp/Datasets/routing/'), 'train-logs-{}'.format(args.data))
_MODELS_DIR = os.path.join(os.path.expanduser('~/Zhong_Exp/Datasets/routing/'), 'models-{}'.format(args.data))
#_DATA_FILE_PATH = os.path.join(_DATA_DIR, _DATA_FILE_NAME)
#_PKL_FILE_PATH = _DATA_FILE_PATH.replace('.dat', '.pkl')

#STAMP = time.strftime('%Y-%m-%d', time.localtime())  # 第二个参数是默认参数

def model_training(capacity_size, cap_train, que_train, ans_train, load):
    ## model parameters
    RNN = layers.LSTM
    rnn_size = 192  #
    hidden_size = 128
    batch_size = 64
    generative_layers = 1
    dropout_rate = 0.2  #
    reduce_patience = 5
    stop_patience = 7  # early stop
    visualize_num = 20
    epochs = 200

    # capacity
    cap_input = layers.Input(shape=(capacity_size,), name='cap_input')  # (*, cap_length)
    cap_output = layers.Dense(hidden_size, activation='relu', name='cap_dense')(cap_input)  # (*, hidden_size)

    # question encoder
    que_input = layers.Input(shape=(maxlen_que, features), name='que_input')  # (*, maxlen_que, features)
    que_output = layers.Bidirectional(RNN(rnn_size // 2, return_sequences=True), name='que_birnn')(que_input)  # (*, maxlen_que, rnn_size)
    que_output = layers.GlobalMaxPool1D(name='que_pooling')(que_output)  # (*, rnn_size)

    # combine capacity & question
    query = layers.concatenate(inputs=[cap_output, que_output])  # (*, hidden_size + rnn_size) -> (*, #)
    query = layers.RepeatVector(maxlen_ans, name='que_repeat')(query)  # (*, maxlen_ans, #)

    # generative part
    for _ in range(generative_layers):
        query = RNN(hidden_size*2, return_sequences=True)(query)
        query = layers.Dropout(dropout_rate)(query)
    query = layers.TimeDistributed(layers.Dense(hidden_size, activation='relu'), name='que_td')(query)
    query = layers.Dropout(dropout_rate, name='que_dropout')(query)
    ans_output = layers.TimeDistributed(layers.Dense(features, activation='softmax'), name='ans_td')(query)  # (*, maxlen_ans, #)
    model = Model(inputs=[cap_input, que_input], outputs=ans_output)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    #plot_model(model, to_file='model.png', show_shapes=True)

    # train the model each generation and show predictions against the validation dataset
    reducer = callbacks.ReduceLROnPlateau(patience=reduce_patience, verbose=1)
    stopper = callbacks.EarlyStopping(patience=stop_patience, verbose=1)
    updater = UpdateMonitor()

    #visualizer = Visualizer(
    #    data=(cap_test, que_test, ans_test),
    #    lookup_table=ntable,
    #    in_graph=graph,
    #    examples=visualize_num,
    #    colored=1-args.redirect,
    #)

    #tester = Logger(test_data=([cap_test, que_test], ans_test))
    plothistory = LossHistory()
    print('start training')
    model.fit([cap_train, que_train], ans_train, batch_size=batch_size, epochs=epochs, verbose=_TRAIN_VERBOSE,
              validation_split=1./8, shuffle=True,
              callbacks=[updater, reducer, stopper, plothistory])
    print('\ntrain finished\n')
    plothistory.loss_plot('epoch')
    loss_file = 'loss' + load + '.pdf'
    loss_location = os.path.join(_MODELS_DIR, loss_file)
    plt.savefig(loss_location, dpi=175)
    _MODELNAME = 'Req2Route' + load + '.h5'
    model_location = os.path.join(_MODELS_DIR, _MODELNAME)
    model.save(model_location)
    print('\nmodel saved!\n')

if __name__ == '__main__':
    # redirect config
    # _STAMP = get_args_info(prefix=NAME, args=args)
    _TRAIN_VERBOSE = 2 if args.redirect else 1
    logfile = None
    stdout_bak = sys.stdout
    print('params:', args)

    KTF.set_session(get_session())

    _LOAD_DIRECT = 'load_direct_model.txt'
    _LOAD_DIRECT_PATH = os.path.join(_DATA_DIR, _LOAD_DIRECT)
    load_lines = []
    with open(_LOAD_DIRECT_PATH, 'r', encoding='gb2312') as fr:
        for load_line in _tqdm(fr, desc='loading'):
            load_line = load_line.strip()
            load_lines.append(load_line)

    for load in load_lines:
        _DATA_FILE_NAME = 'output_BRPC' + load + '.dat'
        DATA_FILE_PATH = os.path.join(_DATA_DIR, _DATA_FILE_NAME)
        PKL_FILE_PATH = DATA_FILE_PATH.replace('.dat', '.pkl')

        if args.redirect:
            try:
                _LOGFILE = 'train' + load + '.log'
                logfile_name = os.path.join(_TRAINLOGS_DIR, _LOGFILE)
                print('training stdout messages will be saved in', logfile_name)
                logfile = open(logfile_name, 'w', encoding='utf8')
                sys.stdout = Flush(logfile)
            except:
                print('create logfile fail, use stdout instead', file=sys.stderr)

        routes, routes_test, min_node_index, max_node_index, min_node_test_index, max_node_test_index = load_data_new(DATA_FILE_PATH, PKL_FILE_PATH, from_pkl=True, save_pkl=False)
        has_congestion, congestion_node, max_inter_capacity_size, has_congestion_test, congestion_node_test, max_inter_capacity_size_test, max_intra_capacity_size_test = data_process(routes, routes_test, min_node_index, max_node_index, min_node_test_index, max_node_test_index)

        if min_node_index == min_node_test_index and max_node_index == max_node_test_index:
            print('Train and test have same distribution')
        else:
            print('ERROR! Train and test different distribution')

        NODES = max_node_index if not has_congestion else congestion_node
        ntable = LookupTable(NODES)

        maxlen_1 = int(np.max([len(route.index_path) for route in routes]))  # longest path length in dataset
        maxlen_2 = int(np.max([len(route_test.index_path) for route_test in routes_test]))  # longest path length in dataset
        if maxlen_1 > maxlen_2:
            maxlen = maxlen_1
        else:
            maxlen = maxlen_2
        print(' - longest path of train and test:', maxlen)
        maxlen_que = 2  # <src> and <dest>
        maxlen_ans = maxlen - maxlen_que  # intermediate nodes in path (answer)
        features = NODES + 1  # add 0 as <padding>

        cap_train, que_train, ans_train = make_train_data(ntable, routes, max_inter_capacity_size, maxlen_que, maxlen_ans, features)
        #cap_test, que_test, ans_test, hid_test = make_test_data(ntable, routes_test, max_inter_capacity_size_test, maxlen_que, maxlen_ans, features, max_intra_capacity_size_test)


        print('building model under load', load, '...')
        model_training(max_inter_capacity_size, cap_train, que_train, ans_train, load)
        print('\ntraining process finished ~~~')

        # restore stdout
        if args.redirect:
            try:
                logfile.close()
                sys.stdout = stdout_bak
            except:
                print('logfile close fail', file=sys.stderr)
