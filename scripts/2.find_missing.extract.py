import pickle
import lang2vec.lang2vec as l2v
import time
import sys
import myutils
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_predict

x, y, names, all_feat_names = pickle.load(open('feats-full.pickle', 'rb'))

all_feat_names = all_feat_names.tolist()

start_time = time.time()

if sys.argv[1] == 'svm':
    from sklearn import svm
    clf = svm.SVC(random_state=myutils.seed, probability=True)
    print('Training svm')
elif sys.argv[1] == 'rf':
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(max_depth=None, n_estimators=100, random_state=myutils.seed, n_jobs=28)
    print('Training random forest')
elif sys.argv[1] == 'logres':
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(random_state=myutils.seed, n_jobs=28, max_iter=500)
    print('Training logistic regression')
else:
    print(sys.argv[1] + ' not implemented')
    exit(1)
# For debugging:
#x = x[:1000]
#y = y[:1000]

pred_y = cross_val_predict(clf, x, y, method='predict_proba', cv=5).tolist()
# Now we are going to collect datapoints for which we have gold
# data, but which were likely to be missing
data_points = []
for i in range(len(pred_y)):
    if y[i] == 1:
        instance = [pred_y[i][0], names[i]]
        data_points.append(instance)

l2v_langs, l2v_data, _ = pickle.load(open('lang2vec.pickle','rb'))

feature_names = l2v.get_features('eng', 'syntax_wals+phonology_wals', header=True)['CODE']

# Add the gold value from the lang2vec database
final_data = []
for item in sorted(data_points):
    lang, feat_name = item[1].split('|')
    feat_idx = feature_names.index(feat_name)
    lang_idx = l2v_langs.index(lang)
    gold_val = l2v_data[lang_idx][feat_idx]
    final_data.append([lang, feat_name, gold_val])

print(len(final_data))
# Save the sorted data, split can then be made on the full 
# data with different sizes. The ones on the bottom should 
# be dev and test.
pickle.dump(final_data, open('final_data.pickle','wb'))



