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
x = []
y = []

def rob(probs):
    mean = sum(probs)/len(probs)
    stdev = statistics.stdev(probs)
    return mean/stdev

def rob2(probs):
    mean = sum(probs)/len(probs)
    stdev = statistics.stdev(probs)
    return stdev/mean

def skew(probs):
    return scipy.stats.skew(probs)
def skew2(probs):
    return scipy.stats.skew(probs, bias=False)

def kursosis(probs):
    return scipy.stats.kurtosis(probs)

def max2(probs):
    return probs[0]



def angle(probs):
    y = list(range(len(probs)))
    angles = []
    for i in range(1, len(probs)):
        angle = (probs[0] - probs[i]) / (y[0]- y[i]) # diff in x / diff in y
        angles.append(angle)
    return max(angles)
    
def angle2(probs):
    y = list(range(len(probs)))
    angles = []
    for i in range(1, len(probs)):
        angle = (probs[0] - probs[i]) / (y[0]- y[i]) # diff in x / diff in y
        angles.append(angle)
    return min(angles)
    
test_scores = {}
test_probs = {}
#test_paths_new = test_paths_new[:5]
for test_path in tqdm(test_paths_new, leave=False):
    test_pred = model_path.replace('model.pt', test_path.split('/')[-2]) + '.out'
    score = getScores(test_path, test_pred)['UPOS'].recall * 100 # Tokens is the other one)
    freqs = read_pos(test_path).values()
    total = sum(freqs)
    probs = sorted([freq/total for freq in freqs], reverse=True)
    test_probs[test_path] = probs
    test_scores[test_path] = score

for method, name in zip([rob, rob2, skew, skew2, kursosis, max, max2, angle, angle2, statistics.stdev], ['rob', 'rob2', 'skew', 'skew2', 'kursosis', 'max', 'max2', 'angle', 'angle2', 'stdev']):
    for test_path in test_paths_new:
        probs = test_probs[test_path]
        score = test_scores[test_path]
        x.append(method(probs)) 
        y.append(score)
    print(name, scipy.stats.pearsonr(x, y))

    ax.scatter(x,y)
    ax.set_xlabel(name)
    ax.set_ylabel('F1')
    #leg = ax.legend()
    #leg.get_frame().set_linewidth(1.5)
    
    fig.savefig('skewdness-' + name + '.pdf', bbox_inches='tight')



