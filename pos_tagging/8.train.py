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

x = []
y = []
for line in open('8.out'):
    tok = line.strip().split('\t')
    y.append(float(tok[1]))
    x.append(tok[3:])
    distr = ast.literal_eval(tok[2])
    for label in labels:
        x[-1].append(distr[label])
    x[-1] = [float(item) for item in x[-1]]


def eval(clf, name):
    pred_y = cross_val_predict(clf, x, y, method='predict', cv=10).tolist()
    total = 0.0
    for gold, pred in zip(y, pred_y):
        total += abs(gold-pred)
    print(name, '{:.2f}'.format(total/len(y)))

clf = RandomForestRegressor(random_state=0, n_estimators=200)
#from sklearn.feature_selection import RFECV
#selector = RFECV(clf, step=1, cv=5)
#selector = selector.fit(x, y)
#print(selector.support_)
#exit(1)
for i in [50,100,200,500,1000]:
   clf = RandomForestRegressor(random_state=0, n_estimators=i)#.fit(x[:split], y[:split])
   eval(clf, 'rf-' + str(i))



for i in [10000, 50000]:
    clf = SGDRegressor(max_iter= i)
    eval(clf, 'SGD-' + str(i))

for i in ['linear', 'rbf']:
    clf = SVR(kernel = i)
    eval(clf, 'SVR-' + i)

#from sklearn.linear_model import ridge_regression
#clf = ridge_regression()
#eval(clf, 'ridge')

from sklearn.linear_model import Lasso
clf = Lasso()
eval(clf, 'lasso')

from sklearn.linear_model import ElasticNet
clf = ElasticNet()
eval(clf, 'elastic')


