
# the main structure is that we get for each cell in the lang2vec matrix
# (language +feature) a gold value (in y), and a list of features describing
# this cell (in x). The features can be based on the language, on the feature,
# or on the text. We will then shuffle them, and do a k-fold prediction (to get
# an idea of performance). In the end we can use confidence thresholds or false
# negatives to get our "likely to be missing, but we have a gold label" cells.

import pickle
import lang2vec.lang2vec as l2v


langs, vectors, vectors_knn = pickle.load(open('lang2vec.pickle', 'rb'))


# Create gold labels for missing values, note that it is of length 
# #langs * #features
# 0 indicates that the feature is missing, 1 indicates that it is present
y = []
x_langs = []
for vector, lang in zip(vectors, langs):
    gold_labels = [0 if val == -100 else 1 for val in vector]
    y.extend(gold_labels)
    x_langs.extend([lang]*len(gold_labels))

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
        wiki_sizes[-1][1] = size
# convert to dict
wiki_sizes = {lang:size for lang, size in wiki_sizes}

# Feature names can be split by '_' and used as features
feature_names = l2v.get_features('eng', 'syntax_wals+phonology_wals', header=True)['CODE']

# Create features
x = []
for lang in x_langs:
    wiki_size = 0 if lang not in wiki_sizes else wiki_sizes[lang] 
    x.append([wiki_size])

# shuffle + k-fold

# train model (we could also save the features, and train/eval models in a
# separate script, might be cleaner)

