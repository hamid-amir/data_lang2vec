
# the main structure is that we get for each cell in the lang2vec matrix
# (language +feature) a gold value (in y), and a list of features describing
# this cell (in x). The features can be based on the language, on the feature,
# or on the text. We will then shuffle them, and do a k-fold prediction (to get
# an idea of performance). In the end we can use confidence thresholds or false
# negatives to get our "likely to be missing, but we have a gold label" cells.

import pickle
import lang2vec.lang2vec as l2v
import time
from tqdm import tqdm
import myutils
import random 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.decomposition import PCA, TruncatedSVD

random.seed(myutils.seed)

langs, vectors, vectors_knn = pickle.load(open('lang2vec.pickle', 'rb'))

n = 100
langs, vectors, vectors_knn = langs[:n], vectors[:n], vectors_knn[:n]
# Feature names can be split by '_' and used as features
feature_names = l2v.get_features('eng', 'syntax_wals+phonology_wals', header=True)['CODE']
print(len(feature_names), len(langs))


# Create gold labels for missing values, note that it is of length 
# #langs * #features
# 0 indicates that the feature is missing, 1 indicates that it is present
y = []
names = []
print('create y and names:')
for vector, lang in tqdm(zip(vectors, langs), total=len(vectors)):
    gold_labels = [0 if val == -100 else 1 for val in vector]
    y.extend(gold_labels)
    for feature in feature_names:
        names.append(lang + '|' + feature)


# load pylogency matrix 
phyl_matrix_sparse = pickle.load(open('phyl-matrix-sparse.pickle', 'rb'))
phyl_matrix = phyl_matrix_sparse.toarray()

# reduce dimension of phylogency by PCA. There are other options like SVD, t-SNE, ... that may worth to try
# I propose to try 10, 30, 50, 70, 90 n_components
# Another idea for dimension reduction: look at this  plt.plot(np.sum(phyl_matrix, axis=0))
pca = PCA(n_components=10)  # Reduce to 100 dimensions
phyl_matrix_pca = pca.fit_transform(phyl_matrix)


# Create features
x = {'lang_id': [], 'feat_id': [], 'geo_lat': [], 'geo_long': [], 'phylogency': [], 'lang_group': [], 'aes_status': [], 'wiki_size': [], 'num_speakers': [], 'lang_fam': [], 'scripts': [], 'feat_name': []}
print('create x(features):')
for langIdx, lang in tqdm(enumerate(langs), total=len(langs)):
    geo = myutils.getGeo(lang)
    phyl = phyl_matrix_pca[langIdx]
    group = myutils.getGroup(lang)
    aes = myutils.getAES(lang)
    wiki_size = myutils.getWikiSize(lang)
    aspj_speakers = myutils.get_aspj_speakers(lang)
    fam = myutils.get_fam(lang)
    scripts = myutils.getScripts(lang)

    for featureIdx, feat_name in enumerate(feature_names):
        instanceIdx = langIdx * len(feature_names) + featureIdx
        # language identifier
        x['lang_id'].append(str(langIdx))

        # Add feature location
        x['feat_id'].append(str(featureIdx))
        
        # latitude and longitude from glottolog
        x['geo_lat'].append(geo[0])
        x['geo_long'].append(geo[1])

        # Pylogency feature from lang2vec
        x['phylogency'].append(phyl)

        # Group from paper
        x['lang_group'].append(group)

        # Group from glottolog
        x['aes_status'].append(int(aes))
        
        # Wikipedia size
        x['wiki_size'].append(wiki_size)

        # Number of speakers
        x['num_speakers'].append(aspj_speakers)

        # Language family is just one string, could have been all hight levels
        # Language family
        x['lang_fam'].append(fam)

        # Scripts
        x['scripts'].append('_'.join(scripts))

        # feature name
        x['feat_name'].append(feat_name)

print('generating done')

def underline_tok(line):
    return line.split('_')

# https://scikit-learn.org/stable/modules/compose.html#columntransformer-for-heterogeneous-data
fam_vectorizer = CountVectorizer(binary=True, tokenizer=underline_tok, ngram_range=(1, 1), analyzer='word')
script_vectorizer = CountVectorizer(binary=True, tokenizer=underline_tok, ngram_range=(1, 1), analyzer='word')
featname_vectorizer = CountVectorizer(binary=True, tokenizer=underline_tok, ngram_range=(1, 1), analyzer='word')

column_trans = ColumnTransformer(
     [('lang_id', OneHotEncoder(dtype='int'), ['lang_id']),
      ('feat_id', OneHotEncoder(dtype='int'), ['feat_id']),
      ('geo_lat', MinMaxScaler(), ['geo_lat']),
      ('geo_long', MinMaxScaler(), ['geo_long']),
      ('lang_fam', fam_vectorizer, 'lang_fam'),
      ('scripts', script_vectorizer, 'scripts'),
      ('feat_name', featname_vectorizer, 'feat_name')],
     remainder='passthrough', verbose_feature_names_out=True)

# why do we need pandas?
import pandas as pd
x = column_trans.fit_transform(pd.DataFrame(x))
all_feat_names = column_trans.get_feature_names_out()
x_numpy = x.toarray()

# shuffle
z = [[feats, gold, name] for feats, gold, name in zip(list(x_numpy), y, names)]
random.shuffle(z)

x = [item[0] for item in z]
y = [item[1] for item in z]
names = [item[2] for item in z]

for i in range(10):
    print(names[i], y[i], x[i][:10])

## k-fold with k=5
#for i in range(5):
#    split = int(len(z) * .8)
#    train_x = x[:split]
#    train_y = y[:split]
#    dev_x = x[split:]
#    dev_y = y[split:]
#    train_names = names[:split]
#    dev_names = names[split:]

#    with open('feats-fold' + str(i) + '.pickle', 'wb') as f:
#        pickle.dump([train_x, train_y, train_names, dev_x, dev_y, dev_names, all_feat_names], f)

# The old 60-20-20 split, is mainly here for legacy reasons
split1 = int(len(z) * .6)
split2 = int(len(z) * .8)
train_x = x[:split1]
dev_x = x[split1:split2]
train_y = y[:split1]
dev_y = y[split1:split2]
train_names = names[:split1]
dev_names = names[split1:split2]

with open('feats.pickle', 'wb') as f:
    pickle.dump([train_x, train_y, train_names, dev_x, dev_y, dev_names, all_feat_names], f)

with open('feats-full.pickle', 'wb') as f:
    pickle.dump([x, y, names, all_feat_names], f)

