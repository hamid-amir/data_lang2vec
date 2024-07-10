import os 
import statistics

import os
from tqdm import tqdm
import myutils
import conll18_ud_eval
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

plt.style.use('scripts/rob.mplstyle')

ud_dir = 'data/ud-treebanks-v2.14/'

test_paths_new = []
test_paths_old = []
for dataset in os.listdir(ud_dir):
    train, dev, test = myutils.getTrainDevTest(os.path.join(ud_dir, dataset))
    if myutils.hasColumn(test, 1):
        if train == '':
            test_paths_new.append(test)
        else:
            test_paths_old.append(test)

class conllFile:
    def __init__(self, path):
        self.data = open(path).readlines()
        self.idx = 0
        self.prev_comment = False

    def readline(self):
        self.idx +=1
        if self.idx != len(self.data):
            tok = self.data[self.idx].split('\t')
            if len(tok) == 10:
                if self.prev_comment:
                    tok[6] = '0'
                    tok[7] = 'root'
                else:
                    tok[6] = '1'
                    tok[7] = 'punct'
                self.prev_comment = False
            else:
                self.prev_comment = True

            return '\t'.join(tok)


def getScores(gold_path, pred_path):
    goldSent = conll18_ud_eval.load_conllu(open(gold_path))
    predSent = conll18_ud_eval.load_conllu(conllFile(pred_path))
    return conll18_ud_eval.evaluate(goldSent, predSent)

fig, ax = plt.subplots(figsize=(8,5), dpi=300)

lm = 'microsoft/infoxlm-large'
diverse = False
smoothing = .5
name = 'UD2.14-pos' + lm.replace('/', '_') + '.' + str(diverse) + '.' + str(smoothing)
model_path = myutils.getModel(name)

if model_path == '':
    print('model not found')
    exit(1)

labels = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']

def read_pos(path):
    data = {label: 0 for label in labels}
    for line in open(path):
        if len(line) > 2 and line[0] != '#':
            pos = line.strip().split('\t')[3]
            data[pos] += 1
    return data

import scipy
import numpy 
test_scores = {}
test_probs = {}
#test_paths_new = test_paths_new[:5]
for test_path in tqdm(test_paths_new, leave=False):
    test_pred = model_path.replace('model.pt', test_path.split('/')[-2]) + '.out'
    score = getScores(test_path, test_pred)['UPOS'].recall * 100 # Tokens is the other one)
    freqs = read_pos(test_path).values()
    total = sum(freqs)
    probs = sorted([freq/total for freq in freqs], reverse=True)
    if score > .7:
        ax.plot(range(len(probs)), probs)

    
ax.set_ylim((0,0.85))
fig.savefig('distributions-good.pdf', bbox_inches='tight')

def read_pos2(path):
    data = {label: 0 for label in labels}
    for line in open(path):
        tok = line.strip().split('\t')
        if len(tok) == 2:
            data[tok[1]] += 1
    return data

data_dir = 'data/MILTALE-CLEAN/'
fig, ax = plt.subplots(figsize=(8,5), dpi=300)
for datafile in tqdm(os.listdir(data_dir)):
    if datafile.endswith('.cleaned'):
        freqs = read_pos2(data_dir + datafile).values()
        total = sum(freqs)
        probs = sorted([freq/total for freq in freqs], reverse=True)
        ax.plot(range(len(probs)), probs)
ax.set_ylim((0,0.85))

fig.savefig('distributions-all.pdf', bbox_inches='tight')


