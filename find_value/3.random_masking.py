
# In this script we only use non-missing (present) instances
# of lang2vec syntax_wals+phonology_wals features of 1720
# languages and create k-fold split for train and test set
# using a RANDOM manner to ready the data for train and evaluate 
# of the predcition classifiers. 
# Important note: we just save names of the features as our folds,
# NOT specificly x and y. This is due to the large size of x.
# You can easily find the corresponding x and y for these names!

import random
import pickle
import lang2vec.lang2vec as l2v
import sys
import os
sys.path.append(os.getcwd())
import scripts.myutils as myutils
from sklearn.model_selection import KFold



random.seed(myutils.seed)

# num_folds
k = 10

# this includes syntax_wals+phonology_wals data for 1720 languages
langs, vectors, vectors_knn = pickle.load(open('lang2vec.pickle', 'rb'))

feature_names = l2v.get_features('eng', 'syntax_wals+phonology_wals', header=True)['CODE']

# select only present values
y = []
names = []
for vector, lang in zip(vectors, langs):
    for val, feature in zip(vector, feature_names):
        if val != -100:
            y.append(val)
            names.append(lang + '|' + feature)

print(f'we have {len(y)} present instances out of total {len(langs) * len(feature_names)} instances.')


# shuffle
z = [[val, name] for val, name in zip(y, names)]
random.shuffle(z)

y = [item[0] for item in z]
names = [item[1] for item in z]



kf = KFold(n_splits=k, shuffle=True, random_state=myutils.seed)
folds = []
for train_idx, test_idx in kf.split(z):
    train_names = [names[i] for i in train_idx]
    test_names = [names[i] for i in test_idx]
    folds.append({'train_names': train_names, 'test_names':test_names})


with open('folds.pickle', 'wb') as f:
    pickle.dump(folds, f)

print(f'{k} folds of present instances were created and saved to the "folds.pickle" file.')
