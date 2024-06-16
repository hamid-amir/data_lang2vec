import time
import sys
import os

from sklearn.base import ClassifierMixin
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_predict

sys.path.append(os.getcwd())
import scripts.myutils as myutils


def get_clf(method: str):
    if method == 'svm':
        from sklearn import svm
        clf = svm.SVC(random_state=myutils.seed, probability=True)
        print('Training svm')
    elif method == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(
            class_weight='balanced_subsample',
            criterion='entropy',
            max_depth=100,
            n_jobs=-1,
            random_state=myutils.seed
        )
        print('Training random forest')
    elif method == 'logres':
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(
            C=10,
            max_iter=500,
            solver='newton-cg',
            tol=0.001,
            n_jobs=-1,
            random_state=myutils.seed
        )
        print('Training logistic regression')
    else:
        raise method + ' not implemented'
    return clf


def cross_validation(clf: ClassifierMixin, x: list, y: list, cv: int = 5):
    start_time = time.time()
    pred_y = cross_val_predict(clf, x, y, method='predict', cv=cv).tolist()
    cors = sum([pred == gold for pred, gold in zip(pred_y, y)])

    base_acc = 100*sum([0 == gold for gold in y])/len(y)
    model_acc = 100 * cors/len(y)
    f1 = 100*f1_score(y, pred_y, pos_label=0)
    recall = 100*recall_score(y, pred_y, pos_label = 0)
    precision = 100*precision_score(y, pred_y, pos_label = 0)
    seconds = time.time() - start_time

    print('Base acc.: {:.2f}'.format(base_acc))
    print('Model acc.: {:.2f}'.format(model_acc))
    print('f1 score 0\'s: {:.2f}'.format(f1))
    print('recall 0\'s: {:.2f}'.format(recall))
    print('precision 0\'s: {:.2f}'.format(precision))
    print('seconds: {:.2f}'.format(seconds))

    return {
        'base_acc': base_acc,
        'model_acc': model_acc,
        'f1': f1,
        'recall': recall,
        'precision': precision,
        'seconds': seconds,
    }