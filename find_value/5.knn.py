import sys
import os
sys.path.append(os.getcwd())

import pickle
import json
from tqdm import tqdm

import scripts.myutils as myutils
from find_missing.utils import cross_validation

# features = [
#     # 'lang_id',
#     'feat_id',
#     # 'geo_lat',
#     # 'geo_long',
#     'lang_group',
#     'aes_status',
#     'wiki_size',
#     'num_speakers',
#     'lang_fam',
#     'scripts',
#     'feat_name',
#     # 'phylogency'
#     # 'inventory_average',
#     # 'phonology_average',
# ]

# myutils.extract_features(
#     'find_value',
#     n_components=100,
#     miltate_n_components=0,
#     # n=100,
#     remove_features=features,
#     miltale_data=False,
#     job_number=0,
#     use_filtered=False
# )

X, Y, names, all_feat_names = pickle.load(open('feats-full_find_value_mil_0.pickle', 'rb'))

results = dict()
for target_feature in tqdm(myutils.target_features):
    # Select feature
    feature_X, feature_Y = list(), list()
    for i, name in enumerate(names):
        if name.split('|')[1] == target_feature:
            feature_X.append(X[i])
            feature_Y.append(Y[i])
    print(len(feature_X), len(feature_Y))

    # remove missing values
    new_X, new_Y = list(), list()
    for x, y in zip(feature_X, feature_Y):
        if min(x) < 0.0:
            continue
        new_X.append(x)
        new_Y.append(y)

    print(len(new_X), len(new_Y))

    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=10, weights='distance')

    result = cross_validation(clf, new_X, new_Y, print_result=False)
    results[target_feature] = result['f1']

with open(f'result/knn.json', 'w') as f:
    f.write(json.dumps(results, indent=4))