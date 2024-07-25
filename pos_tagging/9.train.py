import ast
import statistics

from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import SGDRegressor
from transformers import AutoTokenizer
import ast

labels = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']

def path2feats(path):
    x = []
    y = []
    paths = []
    for line in open(path):
        tok = line.strip().split('\t')
        paths.append(tok[0])
        if tok[1] == 'None':
            y.append(None)
        else:
            y.append(float(tok[1]))
        x.append(tok[3:])
        distr = ast.literal_eval(tok[2])
        for label in labels:
            x[-1].append(distr[label])
        x[-1] = [float(item) for item in x[-1]]
    return x, y, paths

def eval(clf, name):
    pred_y = cross_val_predict(clf, x, y, method='predict', cv=10).tolist()
    total = 0.0
    for gold, pred in zip(y, pred_y):
        total += abs(gold-pred)
    print(name, '{:.2f}'.format(total/len(y)))

x_train, y_train, paths_train = path2feats('8.out')
clf = RandomForestRegressor(random_state=0, n_estimators=200)
clf = RandomForestRegressor(random_state=0, n_estimators=50)#.fit(x[:split], y[:split])
clf.fit(x_train, y_train)

x_new, _, paths_new = path2feats('9.out')
preds = clf.predict(x_new)
for pred, path in zip(preds, paths_new):
    print(pred, path)




 
