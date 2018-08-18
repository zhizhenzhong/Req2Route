import os
import sys
import argparse
import time
import numpy as np

from keras.models import load_model
from common import LookupTable
from misc_utils import Flush
from train_model import load_data_new, graph, data_process, _tqdm

NAME = 'RV4'
parser = argparse.ArgumentParser(prog=NAME)
parser.add_argument('-data', type=int, default=1, help='source of data')
parser.add_argument('-redirect', type=int, choices=(0, 1), default=0, help='redirect stdout to logfile')
args = parser.parse_args()
print('params:', args)
logfile = None
stdout_bak = sys.stdout
STAMP = time.strftime('%Y-%m-%d', time.localtime())  # 第二个参数是默认参数

if args.redirect:
    try:
        logfile_name = 'logs/' + STAMP + '-dynamic-print.log'
        print('dynamic running stdout messages will be saved in', logfile_name)
        logfile = open(logfile_name, 'w', encoding='utf8')
        sys.stdout = Flush(logfile)
    except:
        print('create logfile fail, use stdout instead', file=sys.stderr)

_DATA_DIR = os.path.join(os.path.expanduser('~/datasets/routing'), 'data0524-{}'.format(args.data))


print('===== dynamic test =====')
print('\nloading model...')
model = load_model('models/' + STAMP + '.h5')

routes, routes_test, min_node_index, max_node_index, min_node_test_index, max_node_test_index = load_data_new(from_pkl=True, save_pkl=False)
has_congestion, congestion_node, max_inter_capacity_size, has_congestion_test, congestion_node_test, max_inter_capacity_size_test, max_intra_capacity_size_test = data_process(
    routes, routes_test, min_node_index, max_node_index, min_node_test_index, max_node_test_index)

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

# graph = Graph(topo_file='topo9.yaml')
existing_tasks = set()
# id2seq = dict()  # +1 ed, for window encoder use
id2realseq = dict()  # no +1, for graph decision use

cap_real = np.zeros((1, max_inter_capacity_size), dtype=np.int8)
que_real = np.zeros((1, maxlen_que, features), dtype=np.int8)

count_existing = 0
count_success = 0
count_all = 0

_TEST_FILE_NAME = 'real_time_req.txt'

with open(os.path.join(_DATA_DIR, _TEST_FILE_NAME), 'r', encoding='utf8') as fr, \
        open('logs/' + STAMP + '-dynamic.log', 'w', encoding='utf8') as fw:
    for line in _tqdm(fr, desc='dynamic'):
        fields = line.strip().split()
        print('\n read file:', fields)
        if len(fields) == 5:
            task_id, src, dst, task_type, _ = fields
            task_id, src, dst, task_type = int(task_id), int(src), int(dst), int(task_type)
            print('Current traffic ID:', task_id)
            if task_type == 1:  # build path task
                count_all += 1
                build_success = False
                # get question representation
                que_real[0] = ntable.encode([graph.indexnode[src], graph.indexnode[dst]], maxlen=maxlen_que)

                # update capacity info
                for idx, existing_task_id in enumerate(sorted(existing_tasks)):
                    task_seq = id2realseq.get(existing_task_id, [0])
                    # win_real[0][idx] = ntable.encode(task_seq, maxlen=maxlen)
                    graph.is_buildable(path=task_seq, verbose=False)
                cap_real = graph.cap_mat2list(graph.capacity)  # TODO

                pred_probs = model.predict(x=[cap_real[np.newaxis, :], que_real])
                pred_ans_index = [x - 1 for x in graph.seq_before_zero(pred_probs.argmax(axis=-1)[0])]

                pred_ans_name = [graph.index2node[x] for x in pred_ans_index]
                pred_path_name = [src] + pred_ans_name + [dst]
                print('predict path:', ntable.print_list_path(pred_path_name))
                pred_path_index = [graph.indexnode[x] for x in pred_path_name]
                # check whether it is a good path
                if 61 not in pred_path_index:  # pred a realistic path (no predicted -1)
                    build_success, real_path = graph.is_buildable(pred_path_index, verbose=True)
                    if build_success:
                        count_success += 1
                        count_existing += 1
                        existing_tasks.add(task_id)
                        # id2seq[task_id] = [x for x in pred_path]
                        id2realseq[task_id] = real_path
                        graph.build_path(pred_path_index)
                        print('type{} 1 existing{} task{}: {}'.format(task_type, count_existing, task_id,
                                                                      id2realseq[task_id]), file=fw, flush=True)
                    else:  # build path fail
                        print('type{} 0 existing{} task{}: -'.format(task_type, count_existing, task_id), file=fw,
                              flush=True)
                else:  # also build path fail, since predicted result is -1 already
                    print('type{} 0 existing{} task{}: -'.format(task_type, count_existing, task_id), file=fw,
                          flush=True)

                cap_real = graph.cap_mat2list(graph.capacity)

                if count_all % 1000 == 0:
                    print('\nresults of dynamic check (count_all={}):'.format(count_all))
                    print('- count_success: {} ({:.4f})'.format(count_success, count_success / count_all))
                    print('- count_existing:', count_existing)
            if task_type == 0:  # remove path task
                if task_id in existing_tasks:
                    path_to_break = id2realseq.get(task_id)
                    graph.break_path(path_to_break)
                    existing_tasks.remove(task_id)
                    count_existing -= 1
                    print('type{} -1 existing{} task{}: {}'.format(task_type, count_existing, task_id,
                                                                   id2realseq[task_id]), file=fw, flush=True)
                else:
                    print('type{} 0 existing{} task{}: -'.format(task_type, count_existing, task_id), file=fw,
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
