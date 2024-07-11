import conll18_ud_eval
import os
import myutils
from transformers import AutoTokenizer

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

pos_data = {}
for line in open('swadesh data/all-pos.txt'):
    tok = line.strip().split('\t')
    pos_data[tok[0]] = tok[1]

def lang_data(lang):
    if not os.path.isfile('swadesh data/data/' + lang + '.txt'):
        return {}
    data = {}
    for line in open('swadesh data/data/' + lang + '.txt'):
        tok = line.strip().split(': ')
        if len(tok) != 2:
            continue
        for word in tok[1].split(', '):
            data[word] = pos_data[tok[0]]
    return data


class conllFile:
    def __init__(self, path):
        self.data = open(path).readlines()
        self.idx = 0
        self.prev_comment = False

    def readline(self):
        self.idx +=1
        if self.idx < len(self.data):
            tok = self.data[self.idx].split('\t')
            if len(tok) == 10:
                if self.prev_comment:
                    tok[6] = '0'
                    tok[7] = 'root'
                else:
                    tok[6] = '1'
                    tok[7] = 'punct'
                self.prev_comment = False
                tok[3] = tok[3].split('|')[0].split('=')[0]
            else:
                self.prev_comment = True
            return '\t'.join(tok)


def getScores(gold_path, pred_path):
    goldSent = conll18_ud_eval.load_conllu(open(gold_path))
    predSent = conll18_ud_eval.load_conllu(conllFile(pred_path))
    return conll18_ud_eval.evaluate(goldSent, predSent)

lm = 'microsoft/infoxlm-large'
diverse = False
smoothing = .5
name = 'UD2.14-pos' + lm.replace('/', '_') + '.' + str(diverse) + '.' + str(smoothing)
model_path = myutils.getModel(name)

tokzr = AutoTokenizer.from_pretrained(lm)
if model_path == '':
    print('model not found')
    exit(1)

labels = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']

def read_pos(path):
    
    pos_counts = {label:0 for label in labels}
    confidences = []
    num_subwords = []
    word_lens = []
    totals = 0
    unks = 0
    for line in open(path):
        if len(line) > 2 and line[0] != '#':
            tok = line.strip().split('\t')
            pos = tok[3].split('=')[0]
            pos_counts[pos] += 1
            most_conf = float(tok[3].split('|')[0].split('=')[1])
            confidences.append(most_conf)
            word = tok[1]
            word_lens.append(len(word))
            tok = tokzr.tokenize(word)
            num_subwords.append(len(tok))
            totals += len(tok)
            for subword in tok:
                if subword == tokzr.unk_token:
                    unks += 1
            
    total = sum(pos_counts.values())
    for pos_tag in pos_counts:
        pos_counts[pos_tag] /= total
    
    return pos_counts, sum(confidences)/len(confidences), unks/totals, sum(num_subwords)/len(num_subwords), sum(word_lens)/len(word_lens)


def swadesh(path, lang):
    swadesh_info = lang_data(lang)
    swadesh_overlap = 0
    swadesh_correct = 0
    if swadesh_info != {}:
        for line in open(path):
            tok = line.strip().split('\t')
            if len(tok) == 10:
                word = tok[1]
                pos = tok[3].split('=')[0]
                if word in swadesh_info:
                    swadesh_overlap+= 1
                    if pos == swadesh_info[word]:
                        swadesh_correct += 1
    else:
        print('NoSwadesh', lang, path)
    return swadesh_overlap, swadesh_correct

conv = {}
for line in open('data/iso-639-3.tab'):
    tok = line.strip().split('\t')
    if tok[3] != '':
        conv[tok[3]] = tok[0]

outFile = open('8.out', 'w')
for test_path in test_paths_new:
    test_pred = model_path.replace('model.pt', test_path.split('/')[-2]) + '.out'
    score = getScores(test_path, test_pred)['UPOS'].recall * 100 # Tokens is the other one)
    pos_counts, avg_conf, per_unks, len_subwords, len_words = read_pos(test_pred)
    
    lang = test_path.split('/')[-1].split('_')[0]
    if lang in conv:
        lang = conv[lang]
    swadesh_overlap, swadesh_correct = swadesh(test_pred, lang)
    outFile.write('\t'.join([str(x) for x in [test_path, score, pos_counts, avg_conf, per_unks, len_subwords, len_words, swadesh_overlap, swadesh_correct]]) + '\n')
    print(lang, swadesh_overlap, swadesh_correct)
outFile.close()
