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


def makeGraph(paths, out_path):
    fig, ax = plt.subplots(figsize=(8,5), dpi=300)
    avgs = []
    for lm in tqdm(myutils.lms):
        diverse = False
        smoothing = .5
        name = 'UD2.14-pos' + lm.replace('/', '_') + '.' + str(diverse) + '.' + str(smoothing)
        model_path = myutils.getModel(name)

        if model_path != '':
            scores = []
            for test_path in tqdm(paths, leave=False):
                test_pred = model_path.replace('model.pt', test_path.split('/')[-2]) + '.out'
                scores.append(getScores(test_path, test_pred)['UPOS'].recall * 100) # Tokens is the other one)
            ax.plot(range(len(scores)), sorted(scores, reverse=True), label=lm)
            avgs.append([lm, '{:.2f}'.format(sum(scores)/len(scores))])

    ax.set_xlabel('Number of treebanks')
    ax.set_ylabel('POS recall')
    leg = ax.legend()
    leg.get_frame().set_linewidth(1.5)

    fig.savefig(out_path, bbox_inches='tight')
    for line in avgs:
        print(' & '.join(line) + ' \\\\')
    print()

makeGraph(test_paths_new, 'pos_scores_new.pdf')
makeGraph(test_paths_old, 'pos_scores_seen.pdf')

