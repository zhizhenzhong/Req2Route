import os
import sys
import argparse
import time
import numpy as np

from keras.models import load_model
from common import LookupTable
from misc_utils import Flush
from train_model import load_data_new, Graph, data_process, _tqdm

'''
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

def get_session(gpu_fraction = 0.2):
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = gpu_fraction)
    if num_threads:
        return tf.Session(config = tf.ConfigProto(gpu_options = tpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options = gpu_options))
'''

def get_latency(graph, path):
    latency = 0
    pathlen = len(path)-1
    for i in range(0,pathlen):
        if graph.node2domain[path[i]] != graph.node2domain[path[i+1]]:
            latency = latency + float(graph.interdomain_dis[path[i]][path[i+1]])
    delay = float(latency/200)
    return delay

def get_domains(graph, path):
    domain_num = 1
    pathlen = len(path)-1
    for i in range(0,pathlen):
        if graph.node2domain[path[i]] != graph.node2domain[path[i+1]]:
            domain_num = domain_num + 1
    return domain_num

def get_messageload(graph, path):
    domain_num = 0
    pathlen = len(path)-1
    for i in range(0,pathlen):
        if graph.node2domain[path[i]] != graph.node2domain[path[i+1]]:
            domain_num = domain_num + 1
    message = domain_num*2
    return message

def overlap_cancellation(path):
    cancel_path = path
    for src, dst in zip(path[:-1], path[1:]):
        if src == dst:
            cancel_path.remove(src)
    return cancel_path


NAME = 'RV4'
parser = argparse.ArgumentParser(prog=NAME)
parser.add_argument('-data', type=int, default=1, help='source of data')
parser.add_argument('-redirect', type=int, choices=(0, 1), default=0, help='redirect stdout to logfile')
args = parser.parse_args()
print('params:', args)
logfile = None
stdout_bak = sys.stdout



_DATA_DIR = os.path.join(os.path.expanduser('~/Desktop/routing-test/'), 'data-{}'.format(args.data))
_MODELS_DIR = os.path.join(os.path.expanduser('~/Desktop/routing-test'), 'models-{}'.format(args.data))
_STATICLOGS_DIR = os.path.join(os.path.expanduser('~/Desktop/routing-test'), 'run-logs-{}'.format(args.data))


print('===== dynamic test =====')

#KTF.set_session(get_session())
_LOAD_DIRECT_MODEL = 'load_direct_model.txt'
_LOAD_DIRECT_TEST = 'load_direct_test.txt'
_LOAD_DIRECT_PATH = os.path.join(_DATA_DIR, _LOAD_DIRECT_MODEL)
_LOAD_DIRECT_PATH_TEST = os.path.join(_DATA_DIR, _LOAD_DIRECT_TEST)
model_load_lines = []
test_load_lines = []

with open(_LOAD_DIRECT_PATH, 'r', encoding='gb2312') as fr:
    for load_line in _tqdm(fr, desc='loading'):
        load_line = load_line.strip()
        model_load_lines.append(load_line)

with open(_LOAD_DIRECT_PATH_TEST, 'r', encoding='gb2312') as fr:
    for load_line in _tqdm(fr, desc='loading'):
        load_line = load_line.strip()
        test_load_lines.append(load_line)

for load_train in model_load_lines:
    load_name = 'Req2Route' + load_train + '.h5'
    model_name = os.path.join(_MODELS_DIR, load_name)
    print('\nloading model...' + model_name)
    model = load_model(model_name)

    for load_test in test_load_lines:
        print('Run ' + load_train + 'model on ' + load_test)
        _TEST_FILE_NAME = 'real_time_req' + load_test + '.txt'

        _DATA_FILE_NAME = 'output_BRPC' + load_test + '.dat'
        DATA_FILE_PATH = os.path.join(_DATA_DIR, _DATA_FILE_NAME)
        PKL_FILE_PATH = DATA_FILE_PATH.replace('.dat', '.pkl')

        if args.redirect:
            try:
                logfile_name = 'train' + load_train + '_test' + load_test + '-dynamic-print.log'
                whole_logfile = os.path.join(_STATICLOGS_DIR, logfile_name)
                print('dynamic running stdout messages will be saved in', whole_logfile)
                logfile = open(whole_logfile, 'w', encoding='utf8')
                sys.stdout = Flush(logfile)
            except:
                print('create logfile fail, use stdout instead', file=sys.stderr)

        count_existing = 0
        count_success = 0
        count_all = 0
        routes, routes_test, min_node_index, max_node_index, min_node_test_index, max_node_test_index = load_data_new(DATA_FILE_PATH, PKL_FILE_PATH, from_pkl=True, save_pkl=False)
        has_congestion, congestion_node, max_inter_capacity_size, has_congestion_test, congestion_node_test, max_inter_capacity_size_test, max_intra_capacity_size_test = data_process(routes, routes_test, min_node_index, max_node_index, min_node_test_index, max_node_test_index)

        NODES = max_node_index if not has_congestion else congestion_node
        ntable = LookupTable(NODES)
        maxlen_1 = int(np.max([len(route.index_path) for route in routes]))  # longest path length in dataset
        maxlen_2 = int(
            np.max([len(route_test.index_path) for route_test in routes_test]))  # longest path length in dataset
        if maxlen_1 > maxlen_2:
            maxlen = maxlen_1
        else:
            maxlen = maxlen_2
        print(' - longest path of train and test:', maxlen)
        maxlen_que = 2  # <src> and <dest>
        maxlen_ans = maxlen - maxlen_que  # intermediate nodes in path (answer)
        features = NODES + 1  # add 0 as <padding>

        # graph = Graph(topo_file='topo9.yaml')
        existing_tasks = set()
        # id2seq = dict()  # +1 ed, for window encoder use
        id2realseq = dict()  # no +1, for graph decision use

        cap_real = np.zeros((1, max_inter_capacity_size), dtype=np.int8)
        que_real = np.zeros((1, maxlen_que, features), dtype=np.int8)


        with open(os.path.join(_DATA_DIR, _TEST_FILE_NAME), 'r', encoding='utf8') as fr, \
                open(os.path.join(_STATICLOGS_DIR, 'train' + load_train + '_test' + load_test + '-dynamic.log'), 'w', encoding='utf8') as fw:
            for line in _tqdm(fr, desc='dynamic'):
                fields = line.strip().split()
                print('\n===========')
                print('read file:', fields)
                if len(fields) == 5:
                    task_id, src, dst, task_type, _ = fields
                    task_id, src, dst, task_type = int(task_id), int(src), int(dst), int(task_type)
                    print('Current traffic ID:', task_id)
                    graph = Graph(topo_file='topo9.yaml')
                    if task_type == 1:  # build path task
                        count_all += 1
                        build_success = False
                        # get question representation
                        que_real[0] = ntable.encode([graph.indexnode[src], graph.indexnode[dst]], maxlen=maxlen_que)

                        # update capacity info
                        #for idx, existing_task_id in enumerate(sorted(existing_tasks)):
                            #task_seq = id2realseq.get(existing_task_id, [0])
                            # win_real[0][idx] = ntable.encode(task_seq, maxlen=maxlen)
                            #graph.is_buildable(path=task_seq, verbose=False, use_place=True)
                        cap_real = graph.cap_mat2list(graph.capacity)

                        pred_probs = model.predict(x=[cap_real[np.newaxis, :], que_real])
                        pred_ans_index = [x - 1 for x in graph.seq_before_zero(pred_probs.argmax(axis=-1)[0])]

                        pred_ans_name = [graph.index2node[x] for x in pred_ans_index]
                        pred_path_name = [src] + pred_ans_name + [dst]
                        print('predict path:', ntable.print_list_path(pred_path_name))
                        
                        pred_path_index_ovp = [graph.indexnode[x] for x in pred_path_name]
                        # check whether it is a good path
                        pred_path_index = overlap_cancellation(pred_path_index_ovp)
                        if 61 not in pred_path_index:  # pred a realistic path (no predicted -1)
                            build_success, real_path = graph.is_buildable(pred_path_index, verbose=False, use_place=False)
                            if build_success:
                                count_success += 1
                                count_existing += 1
                                existing_tasks.add(task_id)
                                # id2seq[task_id] = [x for x in pred_path]
                                id2realseq[task_id] = pred_path_index
                                graph.build_path(pred_path_index)
                                print('type{} 1 message {} existing {} task{}: {} latency {}'.format(task_type, get_messageload(graph, pred_path_name), count_existing, task_id, id2realseq[task_id], get_latency(graph, pred_path_name)), file=fw, flush=True)
                                print('setup latency:', get_latency(graph, pred_path_name))
                                print('involved domains:', get_domains(graph, pred_path_name))
                                print('setup incurred inter-domain control message overhead:', get_messageload(graph, pred_path_name))
                            else:  # build path fail
                                print('type{} 0 message {} existing {} task{}: - latency {}'.format(task_type, 0, count_existing, task_id, 0),
                                      file=fw,
                                      flush=True)
                                print('setup latency: 0')
                                print('involved domains: 0')
                                print('setup incurred inter-domain control message overhead: 0')
                        else:  # also build path fail, since predicted result is -1 already
                            print('type{} 0 message {} existing {} task{}: - latency {}'.format(task_type, 0, count_existing, task_id, 0), file=fw,
                                  flush=True)
                            print('setup latency: 0')
                            print('involved domains: 0')
                            print('setup incurred inter-domain control message overhead: 0')

                        cap_real = graph.cap_mat2list(graph.capacity)

                        #if count_all % 1000 == 0:
                            #print('\nresults of dynamic check (count_all={}):'.format(count_all))
                            #print('- count_success: {} ({:.4f})'.format(count_success, count_success / count_all))
                            #print('- count_existing:', count_existing)
                    if task_type == 0:  # remove path task
                        if task_id in existing_tasks:
                            path_to_break = id2realseq.get(task_id)
                            graph.break_path(path_to_break)
                            existing_tasks.remove(task_id)
                            count_existing -= 1
                            path_to_break_name = [graph.index2node[x] for x in path_to_break]
                            print('type{} -1 message {} existing {} task{}: {}'.format(task_type, get_messageload(graph, path_to_break_name), count_existing, task_id, id2realseq[task_id]), file=fw, flush=True)
                        else:
                            print('type{} 0 message {} existing {} task{}: -'.format(task_type, 0, count_existing, task_id, 0), file=fw,
                                  flush=True)
                print('* count_existing:', count_existing)
        print('=' * 30)
        print('\nresults of dynamic check:')
        print('* count_all:', count_all)
        print('* count_success: {} ({:.4f})'.format(count_success, count_success / count_all))
        print('* count_existing:', count_existing)

        if args.redirect:
            try:
                logfile.close()
                sys.stdout = stdout_bak
            except:
                print('logfile close fail', file=sys.stderr)
        print('\ndynamic run process finished ~~~')
