import sys
import os
sys.path.append(os.getcwd())
import scripts.myutils as myutils
import json


ud_dir = 'data/ud-treebanks-v2.14/'

conf_path = 'data/configs/'
if not os.path.isdir('data/configs'):
    os.mkdir('data/configs/')

datasets = {}
for dataset in os.listdir(ud_dir):
    train, dev, test = myutils.getTrainDevTest(os.path.join(ud_dir, dataset))
    if train != '' and myutils.hasColumn(test, 1):
        data_config = {'train_data_path': '../' + train, 'word_idx': 1, 'tasks': {}}
        if dev != '':
            data_config['dev_data_path'] = '../' + dev
        else:
            data_config['dev_data_path'] = '../' + test
        data_config['max_words'] = 200000
        data_config['tasks']['upos'] = {'task_type': 'seq', 'column_idx': 3}
        data_config['tasks']['tok'] = {'task_type': 'tok', 'column_idx': -1, 'pre_split': False}
        datasets[dataset] = data_config
    

json.dump(datasets, open(conf_path + 'ud2.14.json', 'w'), indent=4)

for lm in myutils.lms:
    for diverse in [True, False]:
        config = myutils.load_json('machamp/configs/params.json')
        config['transformer_model'] = lm
        config['batching']['diverse'] = diverse
        config['batching']['sampling_smoothing'] = .5
        config_path = conf_path + lm.replace('/', '_') + '.' + str(config['batching']['diverse']) + '.' + str(config['batching']['sampling_smoothing']) + '.json'
        json.dump(config, open(config_path, 'w'))
        name = 'UD2.14-pos' + lm.replace('/', '_') + '.' + str(config['batching']['diverse']) + '.' + str(config['batching']['sampling_smoothing'])
        cmd = 'python3 train.py --dataset_config ../' + conf_path + 'ud2.14.json --parameters_config ../' + config_path + ' --name ' + name
        if myutils.getModel(name) == '':
            print(cmd)

