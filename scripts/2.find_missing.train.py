import pickle
import time
import sys
import myutils
from sklearn.metrics import f1_score, precision_score, recall_score

train_x, train_y, train_names, dev_x, dev_y, dev_names, all_feat_names = pickle.load(open('feats.pickle', 'rb'))


start_time = time.time()
if sys.argv[1] == 'svm':
    from sklearn import svm
    clf = svm.SVC(random_state=myutils.seed, probability=True)
elif sys.argv[1] == 'rf':
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(max_depth=None, n_estimators=100, random_state=myutils.seed, n_jobs=28)
elif sys.argv[1] == 'logres':
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(random_state=myutils.seed, n_jobs=28, max_iter=500)
else:
    print(sys.argv[1] + ' not implemented')
    exit(1)
clf.fit(train_x, train_y)
#pred_probs = clf.predict_proba(dev_x) # = len(instances) * 2
pred_y = clf.predict(dev_x)
cors = sum([pred == gold for pred, gold in zip(pred_y, dev_y)])
print('Base acc.: {:.2f}'.format(100*sum([0 == gold for gold in dev_y])/len(dev_y)))
print('Model acc.: {:.2f}'.format(100 * cors/len(dev_y)))

print('f1 score 0\'s: {:.2f}'.format(f1_score(dev_y, pred_y, pos_label=0)))
print('recall 0\'s: {:.2f}'.format(recall_score(dev_y, pred_y, pos_label = 0)))
print('precision 0\'s: {:.2f}'.format(precision_score(dev_y, pred_y, pos_label = 0)))
print('seconds: {:.2f}'.format(time.time() - start_time))



# Here is the k-fold implementation, but its kind of slow:
#from sklearn.model_selection import cross_val_score
#clf = svm.SVC(random_state=0)
#print(cross_val_score(clf, x, y, cv=5, scoring='accuracy'))


