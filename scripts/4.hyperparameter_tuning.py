import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_predict

import time
import sys

import myutils

seed = 8446


start_time = time.time()
if sys.argv[1] == 'svm':
    from sklearn import svm
    base_estimator = svm.SVC(random_state=myutils.seed, probability=True)
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': [1, 2, 3, 5, 10],
        'gamma': ['scale', 'auto'],
    }
    print('Training svm')

elif sys.argv[1] == 'rf':
    from sklearn.ensemble import RandomForestClassifier
    base_estimator = RandomForestClassifier(max_depth=None, n_estimators=100, random_state=myutils.seed, n_jobs=28)
    param_grid = {
        'max_depth': [None, 3, 5, 10, 50, 100],
        'min_samples_split': [2, 5, 10, 50, 100],
        'criterion' : ["gini", "entropy", "log_loss"],
        'class_weight' : [None, "balanced", "balanced_subsample"],
        'n_estimators': [50, 100, 200, 500]
    }
    print('Training random forest')

elif sys.argv[1] == 'logres':
    from sklearn.linear_model import LogisticRegression
    base_estimator = LogisticRegression(random_state=myutils.seed, n_jobs=28, max_iter=2000)
    param_grid = {
        'penalty' : ['l1', 'l2', 'elasticnet'],
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'C': [0.001, 0.01, 0.1, 1, 10, 20],
        'solver' : ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
    }
    print('Training logistic regression')

else:
    print(sys.argv[1] + ' not implemented')
    exit(1)


x, y, names, all_feat_names = pickle.load(open('feats-full.pickle', 'rb'))

sh = HalvingGridSearchCV(base_estimator, param_grid, cv=5,
                         factor=50).fit(x, y)

print(sh.best_estimator_)
print(sh.best_score_)

clf = sh.best_estimator_

pred_y = cross_val_predict(clf, x, y, method='predict', cv=5).tolist()
cors = sum([pred == gold for pred, gold in zip(pred_y, y)])

print('Base acc.: {:.2f}'.format(100*sum([0 == gold for gold in y])/len(y)))
print('Model acc.: {:.2f}'.format(100 * cors/len(y)))

print('f1 score 0\'s: {:.2f}'.format(100*f1_score(y, pred_y, pos_label=0)))
print('recall 0\'s: {:.2f}'.format(100*recall_score(y, pred_y, pos_label = 0)))
print('precision 0\'s: {:.2f}'.format(100*precision_score(y, pred_y, pos_label = 0)))

print('seconds: {:.2f}'.format(time.time() - start_time))
