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

def read_pos(path, counts):
    path_counts = {label:0 for label in labels}
    for line in open(path):
        if len(line) > 2 and line[0] != '#':
            pos = line.strip().split('\t')[3]
            path_counts[pos] += 1
    total = sum(path_counts.values())
    for pos_tag in path_counts:
        path_counts[pos_tag] /= total
    return path_counts

import scipy
import numpy 
totals = {label:0 for label in labels}
counts = {label:[] for label in labels}
for test_path in tqdm(test_paths_new, leave=False):
    test_pred = model_path.replace('model.pt', test_path.split('/')[-2]) + '.out'
    score = getScores(test_path, test_pred)['UPOS'].recall * 100 # Tokens is the other one)
    if score > .7:
        path_counts = read_pos(test_path, counts)
        for pos_tag in path_counts:
            totals[pos_tag] += path_counts[pos_tag]
            counts[pos_tag].append(path_counts[pos_tag])

x = []
labels = []
for pos_tag, _ in sorted(totals.items(), key=lambda item: item[1]):
    x.append(counts[pos_tag])
    labels.append(pos_tag)
ax.violinplot(x, showmeans=False, showmedians=True)
ax.set_xticks([y+1.5 for y in range(len(labels))], labels = labels)
plt.xticks(rotation=45,  ha='right')
fig.savefig('distributions-good.pdf', bbox_inches='tight')

def read_pos2(path, counts):
    path_counts = {label:0 for label in labels}
    for line in open(path):
        tok = line.strip().split('\t')
        if len(tok) == 2:
            path_counts[tok[1]] += 1
    total = sum(path_counts.values())
    for pos_tag in path_counts:
        path_counts[pos_tag] /= total
    return path_counts

data_dir = 'data/MILTALE-CLEAN/'
fig, ax = plt.subplots(figsize=(8,5), dpi=300)
totals = {label:0 for label in labels}
counts = {label:[] for label in labels}
for datafile in tqdm(os.listdir(data_dir)):
    if datafile.endswith('.cleaned'):
        path_counts = read_pos2(data_dir + datafile, counts)
        for pos_tag in path_counts:
            totals[pos_tag] += path_counts[pos_tag]
            counts[pos_tag].append(path_counts[pos_tag])
x = []
labels = []
for pos_tag, _ in sorted(totals.items(), key=lambda item: item[1]):
    x.append(counts[pos_tag])
    labels.append(pos_tag)
ax.violinplot(x, showmeans=False, showmedians=True)
ax.set_xticks([y+1.5 for y in range(len(labels))], labels = labels)
plt.xticks(rotation=45,  ha='right')
fig.savefig('distributions-all.pdf', bbox_inches='tight')


