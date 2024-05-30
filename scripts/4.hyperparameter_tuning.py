import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_predict

import time
import sys

import myutils


def hypperparameter_tunning():
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

    clf = sh.best_estimator_
    pred_y = cross_val_predict(clf, x, y, method='predict', cv=5).tolist()
    cors = sum([pred == gold for pred, gold in zip(pred_y, y)])

    base_acc = 100*sum([0 == gold for gold in y])/len(y)
    model_acc = 100 * cors/len(y)
    f1 = 100*f1_score(y, pred_y, pos_label=0)
    recall = 100*recall_score(y, pred_y, pos_label = 0)
    precision = 100*precision_score(y, pred_y, pos_label = 0)
    seconds = time.time() - start_time

    print(sh.best_estimator_)
    print('Base acc.: {:.2f}'.format(base_acc))
    print('Model acc.: {:.2f}'.format(model_acc))
    print('f1 score 0\'s: {:.2f}'.format(f1))
    print('recall 0\'s: {:.2f}'.format(recall))
    print('precision 0\'s: {:.2f}'.format(precision))
    print('seconds: {:.2f}'.format(seconds))

    return {
        'best_model': sh.best_estimator_,
        'base_acc': base_acc,
        'model_acc': model_acc,
        'f1': f1,
        'recall': recall,
        'precision': precision,
        'seconds': seconds,
    }


if __name__ == '__main__':
    from myutils import extract_features
    for n_components in [10, 30, 50, 70, 90]:
        print(f'n_components: {n_components}')
        extract_features(n_components=n_components)
        hypperparameter_tunning()
        print('-'*40)
