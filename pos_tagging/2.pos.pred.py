import myutils
import os
import json

ud_dir = 'data/ud-treebanks-v2.14/'

test_paths = []
for dataset in os.listdir(ud_dir):
    train, dev, test = myutils.getTrainDevTest(os.path.join(ud_dir, dataset))
    if myutils.hasColumn(test, 1):
        test_paths.append(test) 

for lm in myutils.lms:
    diverse = False
    smoothing = .5
    name = 'UD2.14-pos' + lm.replace('/', '_') + '.' + str(diverse) + '.' + str(smoothing)
    model_path = myutils.getModel(name)
    if model_path != '':
        cmd = 'python3 predict.py ../' + model_path + ' ' 
        paths = []
        for test_path in test_paths:
            out_path = model_path.replace('model.pt', test_path.split('/')[2]) + '.out' 
            if not os.path.isfile(out_path + '1'):
                paths.extend(['../' + test_path, '../' + out_path])
        if paths != []:
            cmd += ' '.join(paths) + ' --dataset UD_English-EWT --topn 17'
            print(cmd)

