import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_predict

import time
import sys

seed = 8446


param_grid = {
    # 'max_depth': [None, 3, 5, 10, 50, 100],
    # 'min_samples_split': [2, 5, 10, 50, 100],
    'criterion' : ["gini", "entropy", "log_loss"],
    'class_weight' : [None, "balanced", "balanced_subsample"],
    'n_estimators': [50, 100, 200, 500]
}

# base_estimator = RandomForestClassifier(random_state=seed, n_jobs=28)
start_time = time.time()
base_estimator = RandomForestClassifier(max_depth=None, n_estimators=100, random_state=seed, n_jobs=28)

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
