import pickle
import time
import sys
import os

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

sys.path.append(os.getcwd())
import scripts.myutils as myutils
from find_missing.utils import cross_validation, get_clf


def get_param_grid(method: str):
    if method == 'svm':
        return {
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': [1, 2, 3, 5, 10],
            'gamma': ['scale', 'auto'],
        }
    elif method == 'rf':
        return {
            'max_depth': [None, 3, 5, 10, 50, 100],
            'min_samples_split': [2, 5, 10, 50, 100],
            'criterion' : ["gini", "entropy", "log_loss"],
            'class_weight' : [None, "balanced", "balanced_subsample"],
            'n_estimators': [50, 100, 200, 500]
        }
    elif method == 'logres':
        return {
            'penalty' : ['l1', 'l2', 'elasticnet'],
            'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            'C': [0.001, 0.01, 0.1, 1, 10, 20],
            'solver' : ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
        }


def hypperparameter_tunning():
    method = sys.argv[1]
    base_estimator = get_clf(method)
    param_grid = get_param_grid(method)
    start_time = time.time()

    x, y, names, all_feat_names = pickle.load(open('feats-full_find_missing.pickle', 'rb'))

    sh = HalvingGridSearchCV(base_estimator, param_grid, cv=5,
                            factor=50).fit(x, y)

    clf = sh.best_estimator_

    print(sh.best_estimator_)
    result = cross_validation(clf, x, y)
    result.update({'best_model': sh.best_estimator_})
    seconds = time.time() - start_time
    result['seconds'] = seconds

    return result


if __name__ == '__main__':
    # from myutils import extract_features
    for n_components in [0, 10, 30, 50, 70, 90]:
        print(f'n_components: {n_components}')
        myutils.extract_features('find_missing', n_components=n_components, n=100)
        hypperparameter_tunning()
        print('-'*40)
