import pickle
import time
import sys
import os
sys.path.append(os.getcwd())
import scripts.myutils as myutils
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_predict


x, y, names, all_feat_names = pickle.load(open('feats-full_find_missing.pickle', 'rb'))


start_time = time.time()
if sys.argv[1] == 'svm':
    from sklearn import svm
    clf = svm.SVC(random_state=myutils.seed, probability=True)
    print('Training svm')
elif sys.argv[1] == 'rf':
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(
        class_weight='balanced_subsample',
        criterion='entropy',
        max_depth=100,
        n_jobs=-1,
        random_state=myutils.seed
    )
    print('Training random forest')
elif sys.argv[1] == 'logres':
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
    print(sys.argv[1] + ' not implemented')
    exit(1)

#pred_y = cross_val_predict(clf, x, y, method='predict_proba', cv=5).tolist()
pred_y = cross_val_predict(clf, x, y, method='predict', cv=5).tolist()
cors = sum([pred == gold for pred, gold in zip(pred_y, y)])

print('Base acc.: {:.2f}'.format(100*sum([0 == gold for gold in y])/len(y)))
print('Model acc.: {:.2f}'.format(100 * cors/len(y)))

print('f1 score 0\'s: {:.2f}'.format(100*f1_score(y, pred_y, pos_label=0)))
print('recall 0\'s: {:.2f}'.format(100*recall_score(y, pred_y, pos_label = 0)))
print('precision 0\'s: {:.2f}'.format(100*precision_score(y, pred_y, pos_label = 0)))

print('seconds: {:.2f}'.format(time.time() - start_time))




