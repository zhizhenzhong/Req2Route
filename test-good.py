"""
Grouped boxplots
================

"""
import os
import matplotlib.pyplot as plt
import numpy as np

import numpy as np
import scipy as sp
import scipy.stats

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return h

def _tqdm(iterable, desc=None):
    return iterable

_ORINGINAL_DATA_DIR = os.path.expanduser('~/Zhong_Exp/Datasets/routing')
_DATA_DIR = os.path.join(_ORINGINAL_DATA_DIR, 'test-logs-31')
_RESULT_FILE = 'train_14_1_test_14_1-static-print.log'

data_good_14 = []
data_good_16 = []
data_good_18 = []
data_good_20 = []
data_good_22 = []
data_good_24 = []

data_match_14 = []
data_match_16 = []
data_match_18 = []
data_match_20 = []
data_match_22 = []
data_match_24 = []

error_data_good_14 = []
error_data_good_16 = []
error_data_good_18 = []
error_data_good_20 = []
error_data_good_22 = []
error_data_good_24 = []

error_data_match_14 = []
error_data_match_16 = []
error_data_match_18 = []
error_data_match_20 = []
error_data_match_22 = []
error_data_match_24 = []





for load_train in ['14', '16', '18', '20', '22', '24']:
    locals()['final_good_train_' + load_train] = []
    locals()['error_good_train_' + load_train] = []
    locals()['final_match_train_' + load_train] = []
    locals()['error_match_train_' + load_train] = []
    load = []

    for load_test in ['14', '16', '18', '20', '22', '24']:
        data_good = []
        final_data_good = []
        final_error_good = []
        data_match = []
        final_data_match = []
        final_error_match = []

        load.append(load_train)

        for distribution in ['31', '32', '33', '34']: #different train data distribution
            _DATA_DIR_PATH = _DATA_DIR.replace('31', distribution)
            for train in ['train_'+load_train+'_1', 'train_'+load_train+'_2', 'train_'+load_train+'_3']:
                train_file_name = _RESULT_FILE.replace('train_14_1', train)
                for test in ['test_'+load_test+'_1', 'test_'+load_test+'_2', 'test_'+load_test+'_3']:
                    test_file_name = train_file_name.replace('test_14_1', test)
                    with open(test_file_name, 'r', encoding='gb2312') as fr:
                        for line in _tqdm(fr, desc='loading'):
                            if line.startswith('* count_good:'):
                                current_good = line.split()[3][1:-1]
                                current_good = float(current_good)
                                data_good.append(current_good)

                            if line.startswith('* count_match:'):
                                current_match = line.split()[3][1:-1]
                                current_match = float(current_match)
                                data_match.append(current_match)

        print('good: ', data_good.shape, ', match:', data_match.shape)

        locals()['final_good_train_' + load_train].append(float(sum(data_good) / len(data_good)))
        locals()['error_good_train_' + load_train].append(mean_confidence_interval(data_good, 0.95))
        locals()['final_match_train_' + load_train].append(float(sum(data_match) / len(data_match)))
        locals()['error_match_train_' + load_train].append(mean_confidence_interval(data_match, 0.95))

    locals()['y_'+load_train] = np.array(locals()['final_good_train_' + load_train])
    locals()['y_err_'+load_train] = np.array(locals()['error_good_train_' + load_train])
    locals()['z_'+load_train] = np.array(locals()['final_match_train_' + load_train])
    locals()['z_err_'+load_train] = np.array(locals()['error_match_train_' + load_train])


x = np.array(load)


plt.figure()


n_groups = 6
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.4
error_config = {'ecolor': '0.3'}

for load_train in ['14', '16', '18', '20', '22', '24']:
    locals()['rects1'+load_train] = plt.bar(x, locals()['y_'+load_train], bar_width,
                                            alpha=opacity,
                                            color='b',
                                            yerr=locals()['y_err_'+load_train] ,
                                            error_kw=error_config,
                                            label='Men')


# draw temporary red and blue lines and use them to create a legend

plt.legend(loc='lower right')
plt.grid(True)
plt.yscale("log")
plt.xlabel("Traffic load per node on testing data(Erlang)")
plt.ylabel("Routing accuracy")


plt.xlim(13, 25)
plt.ylim(0.8, 1)
plt.tight_layout()
plt.savefig('blocking.pdf', dpi=175)


