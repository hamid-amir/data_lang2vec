import myutils
import os
import json

ud_dir = 'data/ud-treebanks-v2.14/'

conf_path = 'data/configs/'
if not os.path.isdir('data/configs'):
    os.mkdir('data/configs/')

datasets = {}
for dataset in os.listdir(ud_dir):
    train, dev, test = myutils.getTrainDevTest(os.path.join(ud_dir, dataset))
    if not myutils.hasColumn(test, 1):
        continue
    all_path = test.replace('test.conllu', 'all.conllu')
    if not os.path.isfile(all_path):
        all_data = []
        if train != '':
            all_data.extend(open(train).readlines())
        if dev != '':
            all_data.extend(open(dev).readlines())
        if test != '':
            all_data.extend(open(test).readlines())
        if all_data == []:
            continue
        with open(all_path, 'w') as all_out:
            [all_out.write(line) for line in all_data]

    data_config = {'train_data_path': '../' + all_path, 'word_idx': 1, 'tasks': {}}
    data_config['max_words'] = 100000
    data_config['tasks']['upos'] = {'task_type': 'seq', 'column_idx': 3}
    data_config['tasks']['tok'] = {'task_type': 'tok', 'column_idx': -1, 'pre_split': False}
    datasets[dataset] = data_config
    

data_config_path = 'data/configs/ud2.14.all.json'
json.dump(datasets, open(data_config_path, 'w'), indent=4)

lm = 'microsoft/infoxlm-large'
diverse = False
smoothing = .5
param_config_path = 'data/configs/microsoft_infoxlm-large.True.0.5.json'
name = 'UD2.14-pos' + lm.replace('/', '_') + '.' + str(diverse) + '.' + str(smoothing) + '.all'
cmd = 'python3 train.py --dataset_config ../' + data_config_path +' --parameters_config ../' + param_config_path + ' --name ' + name
if myutils.getModel(name) == '':
    print(cmd)

