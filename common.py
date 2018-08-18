"""
Common module for Various Models

Created on 2017.04.28
"""
import itertools
import time

import numpy as np
from keras.callbacks import Callback

class _Colors:
    """for better visualization"""
    Good = '\033[92m'
    Bad = '\033[91m'
    Match = '\033[95m'
    Close = '\033[0m'


class Route:
    """current route path and its histories (other existing paths)"""
    def __init__(self, index_path=None, inter_capacity=None, intra_capacity = None, histories=None, kpaths=None):
        self.index_path = index_path if index_path else []  # shortest path, index path
        self.inter_capacity = inter_capacity if inter_capacity else []
        self.intra_capacity = intra_capacity if intra_capacity else []
        self.histories = histories if histories else []
        self.kpaths = kpaths if kpaths else []  # k shortest path

class LookupTable:
    """encode and decode between node/path and its neural encoding"""
    def __init__(self, features):
        self.features = features  # usually means how many nodes

    def encode(self, seq, maxlen):
        """
        encode single path
        :param seq: one single path, e.g., path=[2, 3, 1, 5] (features=5, i.e., 5 nodes)
        :param maxlen: max length of all possible paths
        :param onehot: use one-hot or not
        :return: encoded path, e.g., [[0, 0, 1, 0, 0, 0], ..., [0, 0, 0, 0, 0, 1], ...] (one-hot) or [2, 3, 1, 5, 0, ...]
        """
        add_seq = [x + 1 for x in seq]
        padded_seq = add_seq + [0] * (maxlen - len(add_seq))  # post-paddding, make all routes equal length

        x = np.zeros((maxlen, self.features + 1))  # 0 is used for padding, so add one more feature
        for i, node in enumerate(padded_seq):
            #print('seq:', seq, i, node)
            #print(i, node)
            x[i, node] = 1  #vectorize a node index into a binary vector
        return x

    @staticmethod
    def decode(seq, calc_argmax=True, return_str=True):
        """
        decode single path
        """
        if calc_argmax:
            minus_seq = [x for x in seq.argmax(axis=-1)]  # decode process, ndim: 2 -> 1
        # remove padded zeros
        zero_at = -1
        for i, node in enumerate(minus_seq):  # e.g., seq = [2, 3, 1, 5, 0, 0, 0]
            if node == 0:
                zero_at = i
                break
        seq_before_zero = minus_seq[:zero_at] if zero_at > -1 else minus_seq
        if return_str:
            return '-'.join(str(x-1) for x in seq_before_zero) if len(seq_before_zero) > 0 else '0'
        else:
            return minus_seq

    def print_list_path(self, seq):
        return '-'.join(str(x) for x in seq)


class Visualizer(Callback):
    def __init__(self, data, lookup_table, in_graph, examples=10, colored=True):
        """
        Visualize examples in test dataset
        :param data: validation dataset, format: [aux_input_1, ..., aux_input_n, question_input, answer_output]
        :param lookup_table: nodes lookup table
        """
        self.data = data
        #self.congestion = congestion
        self.lookup_table = lookup_table
        self.in_graph = in_graph
        self.examples = int(np.min((examples, len(data[0]))))
        self.colored = colored
        assert hasattr(lookup_table, 'decode')
        super(Visualizer, self).__init__()

    def on_epoch_end(self, epoch, in_graph, logs=None):
        indices = np.random.choice(range(len(self.data[0])), self.examples, replace=False)
        count_all, count_good = 0, 0

        #_minus1 = lambda x: -1 if (self.congestion[0] and x == self.congestion[1]) else x - 1
        for idx in indices:
            if type(self.data[-1]) is list:  # multi outputs (k shortest path)
                count_all += len(self.data[-1])
                inputs = [x_val[np.array([idx])] for x_val in self.data[0:-1]]
                outputs = [data[np.array([idx])] for data in self.data[-1]]
                preds = self.model.predict(inputs, verbose=0)

                query_seq = self.lookup_table.decode(inputs[-1][0])
                query_seq_name = self.in_graph.get_route_name(query_seq)
                correct_seqs = [self.lookup_table.decode(output[0]) for output in outputs]
                correct_seqs_name = self.in_graph.get_route_name(correct_seqs)

                guess_seq_array = [x for x in self.in_graph.seq_before_zero(preds.argmax(axis=-1)[0])]
                print(guess_seq_array)
                guess_array = [query_seq[0] + guess_seq_array + query_seq[-1]]
                print(guess_array)

                print('-' * 15)
                print('- LinkState:', )
                print('- Q:', self.lookup_table.print_list_path(query_seq_name))
                print('- A:', ', '.join(str(x) for x in correct_seqs_name))
                print('- P:', end=' ')
                for index, (correct_seq, guess_seq) in enumerate(zip(correct_seqs, guess_seqs)):
                    end_str = '\n' if index == len(self.data[-1]) - 1 else ', '
                    match_success, match_success_path = self.in_graph.is_buildable(guess_array)
                    if correct_seq == guess_seq:
                        count_good += 1
                        if self.colored:
                            print(_Colors.Good + guess_seq + _Colors.Close, end=end_str)
                        else:
                            print(guess_seq, end=end_str)
                    elif match_success:
                        if self.colored:
                            print(_Colors.Bad + guess_seq + _Colors.Close, end=end_str)
                        else:
                            print(guess_seq, end=end_str)
                    else:
                        if self.colored:
                            print(_Colors.Bad + guess_seq + _Colors.Close, end=end_str)
                        else:
                            print(guess_seq, end=end_str)

            else:  # single output (shortest path)
                count_all += 1
                #linkstate = [p[np.array(idx)] for p in self.data[0]]
                inputs = [x_val[np.array([idx])] for x_val in self.data[0:-1]]
                output = self.data[-1][np.array([idx])]
                preds = self.model.predict(inputs, verbose=0)

                linkcaps = inputs[0][0]
                query_seq = self.lookup_table.decode(inputs[-1][0])
                correct_seq = self.lookup_table.decode(output[0])
                guess_seq = self.lookup_table.decode(preds[0])

                query_seq_name = self.lookup_table.print_list_path(self.in_graph.get_route_name(query_seq))
                correct_seq_name = self.lookup_table.print_list_path(self.in_graph.get_route_name(correct_seq))
                guess_seq_name = self.lookup_table.print_list_path(self.in_graph.get_route_name(guess_seq))

                print('-' * 15)
                print('- LinkStates:', linkcaps)
                print('- Q (index):', query_seq)
                print('- Q:', query_seq_name)
                print('- A (index):', correct_seq)
                print('- A:', correct_seq_name)
                print('guess (index):', guess_seq)
                if correct_seq == guess_seq:
                    count_good += 1
                    if self.colored:
                        print(_Colors.Good + '- Y: ' + guess_seq_name + _Colors.Close)
                    else:
                        print('- Y: ' + guess_seq_name)
                else:
                    #que_array = [x for x in self.data[1][np.array([idx])].argmax(axis=-1)[0]]
                    #guess_seq_array = [x for x in self.in_graph.seq_before_zero(preds.argmax(axis=-1)[0])]

                    query_list = [x for x in query_seq.split('-')]
                    guess_list = [x for x in guess_seq.split('-')]
                    whole_guess_list = [query_list[0]] + guess_list + [query_list[-1]]
                    #whole_guess_list_seq = '-'.join(str(x-1) for x in whole_guess_list)

                    #whole_guess_array_name = self.in_graph.get_route_name(guess_array_seq)
                    #print('whole guess:', self.lookup_table.print_list_path(whole_guess_array_name))

                    if 61 not in whole_guess_list:
                        match_success, match_success_path = self.in_graph.is_buildable(whole_guess_list, verbose=True)

                    if match_success:
                        if self.colored:
                            print(_Colors.Match + '- M: ' + guess_seq_name + _Colors.Close)
                        else:
                            print('- M: ' + guess_seq_name)
                    else:
                        if self.colored:
                            print(_Colors.Bad + '- N: ' + guess_seq_name + _Colors.Close)
                        else:
                            print('- N: ' + guess_seq_name)

                #current_capacity = self.routes_test
                #print('- Capacity:' + current_capacity)
        print('visualize summary: {}/{} ({:.4f})'.format(count_good, count_all, count_good/count_all))


if __name__ == '__main__':
    pass

