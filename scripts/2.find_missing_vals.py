
# the main structure is that we get for each cell in the lang2vec matrix
# (language +feature) a gold value (in y), and a list of features describing
# this cell (in x). The features can be based on the language, on the feature,
# or on the text. We will then shuffle them, and do a k-fold prediction (to get
# an idea of performance). In the end we can use confidence thresholds or false
# negatives to get our "likely to be missing, but we have a gold label" cells.

import pickle
import lang2vec.lang2vec as l2v
import time

langs, vectors, vectors_knn = pickle.load(open('lang2vec.pickle', 'rb'))

# Feature names can be split by '_' and used as features
feature_names = l2v.get_features('eng', 'syntax_wals+phonology_wals', header=True)['CODE']
print(len(feature_names), len(langs))


# Create gold labels for missing values, note that it is of length 
# #langs * #features
# 0 indicates that the feature is missing, 1 indicates that it is present
y = []
names = []
for vector, lang in zip(vectors, langs):
    gold_labels = [0 if val == -100 else 1 for val in vector]
    y.extend(gold_labels)
    for feature in feature_names:
        names.append(lang + '|' + feature)

# Read wikipedia sizes
two2three = {}
threes = set()
lang2code = {}
for line in open('scripts/iso-639-3.tab').readlines()[1:]:
    tok = line.strip().split('\t')
    if tok[3] != '':
        two2three[tok[3]] = tok[0]
    threes.add(tok[0])
    lang2code[tok[6]] = tok[0]
wiki_sizes = []
for line in open('scripts/List_of_Wikipedias'):
    if line.startswith('<td><a href="https://en.wikipedia.org/wiki/') and '_language' in line:
        lang = line.strip().split('<')[-3].split('>')[-1]
    if line.startswith('<td><bdi lang='):
        lang_code = line.strip().split('"')[1].split('-')[0]
        if lang_code in two2three:
            wiki_sizes.append([two2three[lang_code], 0])
        elif lang_code in threes:
            wiki_sizes.append([lang_code, 0])
        elif lang in lang2code:
            wiki_sizes.append([lang2code[lang], 0])
    if 'rg/wiki/Special:Statistics" class="extiw" title=' in line:
        size = line.strip().split('>')[2].split('<')[0]
        wiki_sizes[-1][1] = int(size.replace(',',''))
# convert to dict
wiki_sizes = {lang:size for lang, size in wiki_sizes}


# Create features
x = []

# start with wikipedia size
for lang in langs:
    wiki_size = 0 if lang not in wiki_sizes else wiki_sizes[lang] 
    x.extend([[wiki_size] for _ in range(len(feature_names))])

# add feature location (as binary) feature
for langIdx, _ in enumerate(langs):
    for featureIdx, _ in enumerate(feature_names):
        instanceIdx = langIdx * len(feature_names) + featureIdx
        # convert to binary
        features_to_add = [0] * len(feature_names)
        features_to_add[featureIdx] = 1
        x[instanceIdx].extend(features_to_add)
        
# shuffle + k-fold
z = [[feats, gold, name] for feats, gold, name in zip(x, y, names)]
import random
random.seed(8446)
random.shuffle(z)

x = [item[0] for item in z]
y = [item[1] for item in z]
names = [item[2] for item in z]

for i in range(100):
    print(names[i], y[i], x[i][:10])

split1 = int(len(z) * .6)
split2 = int(len(z) * .8)
train_x = x[:split1]
dev_x = x[split1:split2]
train_y = y[:split1]
dev_y = y[split1:split2]
train_names = names[:split1]
dev_names = names[split1:split2]
print('base')
print(len(dev_y), sum([0 == gold for gold in dev_y]))

start_time = time.time()
from sklearn import svm
# train model (we could also save the features, and train/eval models in a
# separate script, might be cleaner)
clf = svm.SVC(random_state=8446, probability=True)
clf.fit(train_x, train_y)
#pred_probs = clf.predict_proba(dev_x) # = len(instances) * 2
pred_y = clf.predict(dev_x)
print('svc')
print(len(dev_y), sum([pred == gold for pred, gold in zip(pred_y, dev_y)]))
print("--- %s seconds ---" % (time.time() - start_time))


start_time = time.time()
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=8446)
clf.fit(train_x, train_y)
pred_y = clf.predict(dev_x)
print('logres')
print(len(dev_y), sum([pred == gold for pred, gold in zip(pred_y, dev_y)]))
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=None, num_estimators=100, random_state=0)
clf.fit(train_x, train_y)
pred_y = clf.predict(dev_x)
print('randomforest')
print(len(dev_y), sum([pred == gold for pred, gold in zip(pred_y, dev_y)]))
print("--- %s seconds ---" % (time.time() - start_time))

# Here is the k-fold implementation, but its kind of slow:
#from sklearn.model_selection import cross_val_score
#clf = svm.SVC(random_state=0)
#print(cross_val_score(clf, x, y, cv=5, scoring='accuracy'))


