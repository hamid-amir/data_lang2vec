import pickle
import time
import sys

train_x, train_y, train_names, dev_x, dev_y, dev_names = pickle.load(open('feats.pickle', 'rb'))

print('base')
print(len(dev_y), sum([0 == gold for gold in dev_y]))

start_time = time.time()
if sys.argv[1] == 'svm':
    from sklearn import svm
    clf = svm.SVC(random_state=8446, probability=True)
elif sys.argv[1] == 'rf':
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(max_depth=None, n_estimators=100, random_state=8446)
elif sys.argv[1] == 'logres':
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(random_state=8446)
else:
    print(sys.argv[1] + ' not implemented')
    exit(1)
clf.fit(train_x, train_y)
#pred_probs = clf.predict_proba(dev_x) # = len(instances) * 2
pred_y = clf.predict(dev_x)
print(len(dev_y), sum([pred == gold for pred, gold in zip(pred_y, dev_y)]))
print("--- %s seconds ---" % (time.time() - start_time))



# Here is the k-fold implementation, but its kind of slow:
#from sklearn.model_selection import cross_val_score
#clf = svm.SVC(random_state=0)
#print(cross_val_score(clf, x, y, cv=5, scoring='accuracy'))


