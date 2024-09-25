import sys
import os
sys.path.append(os.getcwd())
import pickle
import json
from tqdm import tqdm
import scripts.myutils as myutils
from find_missing.utils import cross_validation
from sklearn.metrics import accuracy_score, f1_score


# original lang2vec KNN approach used only these features: lang_id, phylogency, inventory_average, and geo
remove_features = [
    # 'lang_id',
    'feat_id',
    'geo_lat',
    'geo_long',
    'lang_group',
    'aes_status',
    'wiki_size',
    'num_speakers',
    'lang_fam',
    'scripts',
    'feat_name',
    # 'phylogency',
    # 'inventory_average',
    'phonology_average',
    # 'geo',
]



# Construct dataset using all langs. We will train and test on them using k-fold cross validation which is our first validation approach (similar to the original lang2vec)
myutils.extract_features(
    classifier='find_value',
    remove_features=remove_features,
    dimension_reduction_method=None,  # To have all the components of phylogency
)
X, Y, names, all_feat_names = pickle.load(open('feats-full_find_value_0.pickle', 'rb'))


# extract feats that are most probable to be missing. We will use them as our second validation approach 
probable_missings = pickle.load(open('final_data.pickle', 'rb'))
probable_missings = probable_missings[-int(len(probable_missings)*0.2):]   # select top 20% of the most probable missing ones
probable_missings_langs_feats = [(pm[0],pm[1]) for pm in probable_missings]   # [(lang, feat)]

results = dict()
results_pm = dict()

for target_feature in tqdm(myutils.target_features):
# for target_feature in ['S_VOX', 'S_OBLIQUE_AFTER_VERB', 'P_COMPLEX_ONSETS', 'P_NASALS', 'S_SUBJECT_BEFORE_VERB']:
    feature_X, feature_Y = list(), list()
    feature_X_pm, feature_Y_pm = list(), list()
    X_test, Y_test = list(), list()
    langs, langs_test = list(), list()
    for i, name in enumerate(names):
        lang, feat = name.split('|')[0], name.split('|')[1]
        if feat == target_feature:
            feature_X.append(X[i])
            feature_Y.append(Y[i])
            langs.append(lang)
        if feat == target_feature and (lang, feat) not in probable_missings_langs_feats:
            feature_X_pm.append(X[i])
            feature_Y_pm.append(Y[i])
        if feat == target_feature and (lang, feat) in probable_missings_langs_feats:
            X_test.append(X[i])
            Y_test.append(Y[i])
            langs_test.append(lang)


    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=10, weights='distance')

    # First eval : k-fold cross validation using all present values: k=10 according to the original lang2vec paper
    result = cross_validation(clf, feature_X, feature_Y, print_result=False, cv=min(10, len(feature_X)))
    results[target_feature] = {
        'f1': result['f1'],
        'predictions': [] 
    }
    for i, (true, pred) in enumerate(zip(feature_Y, result['pred_y'])):
        results[target_feature]['predictions'].append({
            'is_correct': 'correct' if true == pred else 'incorrect',
            'true_label': true,
            'predicted_label': pred,
            'language': langs[i],
            'lang_geo': myutils.getGeo(langs[i]),
            'lang_group': myutils.getGroup(langs[i]),
            'lang_aes': myutils.getAES(langs[i]),
            'lang_wikiSize': myutils.getWikiSize(langs[i]),
            'lang_speakers': myutils.get_aspj_speakers(langs[i]),
            'lang_fam': myutils.get_fam(langs[i]),
            'lang_scripts': list(myutils.getScripts(langs[i])),
        })


    # Second eval : probable missings validation
    if len(X_test) > 0:
        clf.fit(feature_X_pm, feature_Y_pm)
        y_pred = clf.predict(X_test)
        f1 = 100*f1_score(Y_test, y_pred, pos_label=0)
        results_pm[target_feature] = {
            'f1': f1,
            'predictions': []  
        }

        for i, (true, pred) in enumerate(zip(Y_test, y_pred)):
            results_pm[target_feature]['predictions'].append({
                'is_correct': 'correct' if true == pred else 'incorrect',
                'true_label': true,
                'predicted_label': pred,
                'language': langs_test[i],
                'lang_geo': myutils.getGeo(langs_test[i]),
                'lang_group': myutils.getGroup(langs_test[i]),
                'lang_aes': myutils.getAES(langs_test[i]),
                'lang_wikiSize': myutils.getWikiSize(langs_test[i]),
                'lang_speakers': myutils.get_aspj_speakers(langs_test[i]),
                'lang_fam': myutils.get_fam(langs_test[i]),
                'lang_scripts': list(myutils.getScripts(langs_test[i])),
            })




with open(f'result/knn.json', 'w') as f:
    f.write(json.dumps(results, indent=4))

with open(f'result/knn_pm.json', 'w') as f:
    f.write(json.dumps(results_pm, indent=4))