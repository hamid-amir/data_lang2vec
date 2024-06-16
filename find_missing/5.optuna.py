import pickle
import time
import sys
import os

import optuna
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

sys.path.append(os.getcwd())
import scripts.myutils as myutils
from find_missing.utils import cross_validation, get_clf


def objective(trial, method: str, x, y):
    if method == 'svm':
        C = trial.suggest_loguniform('C', 1e-4, 1e3)
        kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
        degree = trial.suggest_int('degree', 1, 10)
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
        clf = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma)
    elif method == 'rf':
        max_depth = trial.suggest_int('max_depth', 3, 100)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 100)
        criterion = trial.suggest_categorical('criterion', ["gini", "entropy", "log_loss"])
        class_weight = trial.suggest_categorical('class_weight', [None, "balanced", "balanced_subsample"])
        n_estimators = trial.suggest_int('n_estimators', 50, 500)
        clf = RandomForestClassifier(max_depth=max_depth, min_samples_split=min_samples_split, 
                                     criterion=criterion, class_weight=class_weight, n_estimators=n_estimators)
    elif method == 'logres':
        penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
        tol = trial.suggest_loguniform('tol', 1e-5, 1e-1)
        C = trial.suggest_loguniform('C', 1e-4, 1e3)
        solver = trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'])
        clf = LogisticRegression(penalty=penalty, tol=tol, C=C, solver=solver)

    result = cross_validation(clf, x, y, print_result=False)

    return result['f1']


def hyperparameter_tuning():
    method = sys.argv[1]
    start_time = time.time()

    x, y, names, all_feat_names = pickle.load(open('feats-full_find_missing.pickle', 'rb'))

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, method, x, y), n_trials=100)

    best_trial = study.best_trial
    print(f'Best trial: {best_trial.values}')

    clf = get_clf(method).set_params(**best_trial.params)
    clf.fit(x, y)
    result = cross_validation(clf, x, y)
    result.update({'best_model': clf})
    seconds = time.time() - start_time
    result['seconds'] = seconds

    return result


if __name__ == '__main__':
    for n_components in [0, 10, 30, 50, 70, 90]:
        print(f'n_components: {n_components}')
        myutils.extract_features('find_missing', n_components=n_components, n=100)
        hyperparameter_tuning()
        print('-'*40)
