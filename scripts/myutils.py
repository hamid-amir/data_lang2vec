import os
import json
import pickle
import lang2vec.lang2vec as l2v
import time
from tqdm import tqdm
from typing import Literal
import random 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.decomposition import PCA, TruncatedSVD


lms = ['bert-base-multilingual-cased', 'cardiffnlp/twitter-xlm-roberta-base', 'microsoft/infoxlm-large', 'microsoft/mdeberta-v3-base', 'studio-ousia/mluke-large', 'xlm-roberta-large', 'facebook/xlm-roberta-xl', 'Twitter/twhin-bert-large' ]  


seed = 8446
random.seed(seed)

# Handling of iso639 codes
iso639 = {}
macros = set()
for line in open('data/iso-639-3.tab', encoding="utf8").readlines()[1:]:
    tok = line.strip().split('\t')
    if tok[4] == 'I':
        iso639[tok[0]] = tok[6]
    elif tok[4] == 'M':
        macros.add(tok[0])

iso639_conv = {}
for line in open('data/iso-639-3_Retirements.tab', encoding="utf8").readlines()[1:]:
    tok = line.strip().split('\t')
    prev = tok[0]
    new = tok[3]
    if new != '':
        iso639_conv[prev] = new

def code2iso(code):
    if code in iso639_conv:
        code = iso639_conv[code]

    if code in iso639: 
        return code
    else:
        return None
    #elif code in macros:
    #    return "MACRO"
    #elif '-' in code:
    #    return "DIALECT"
    #else:
    #    return None


# Handling of language families of glottolog
trees = []
from newick import loads
for line in open('data/tree_glottolog_newick.txt', encoding="utf8"):
    tree = loads(line.strip())
    trees.append(tree[0])

def getPath(langCode):
    found = None
    for tree in trees:
        if found != None:
            break
        for item in tree.walk():
            if '[' + langCode + ']' in item.name:
                found = item
    if found == None:
        return None
    curNode = found
    path = [curNode.name]
    while hasattr(curNode, 'ancestor') and curNode != None:
        curNode = curNode.ancestor
        if curNode == None:
            break
        path.append(curNode.name)
    return path

def get_fam(lang):
    found = None
    for tree in trees:
        if found != None:
            break
        for item in tree.walk():
            if '[' + lang + ']' in item.name:
                found = item
    if found == None:
        return ''
    curNode = found
    return curNode.name.split('[')[1].split(']')[0]




# Reading wikipedia sizes. They use two character language codes, 
# so we also need the conversion
two2three = {}
threes = set()
lang2code = {}
for line in open('data/iso-639-3.tab', encoding="utf8").readlines()[1:]:
    tok = line.strip().split('\t')
    if tok[3] != '':
        two2three[tok[3]] = tok[0]
    threes.add(tok[0])
    lang2code[tok[6]] = tok[0]
wiki_sizes = []
for line in open('data/List_of_Wikipedias', encoding="utf8"):
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
        wiki_sizes[-1][1] = int(size.replace(',', ''))
# convert to dict
wiki_sizes = {lang:size for lang, size in wiki_sizes}
def getWikiSize(lang):
    if lang in wiki_sizes:
        return wiki_sizes[lang]
    return 0

# Read glotscript data. Note that Brai (braille) is removed, because
# annotation is incomplete
glotscript_data = {}
for line in open('data/GlotScript.tsv', encoding="utf8").readlines()[1:]:
    tok = line.strip().lower().split('\t')
    if len(tok) > 1:
        glotscript_data[tok[0]] = set([x.strip() for x in tok[1].split(',') if x != ' brai'])
def getScripts(lang):
    if lang in glotscript_data:
        return glotscript_data[lang]
    else:
        return []

group2name = {'0': '0. The Left-Behinds', '1': '1. The Scraping-Bys', '2': '2. The Hopefuls', '3': '3. The Rising Stars', '4': '4. The Underdogs', '5': '5. The Winners'}
iso2group = {}
for line in open('data/lang2tax.txt.codes', encoding="utf8"):
    iso, group, name = line.strip().split(',') 
    iso2group[iso] = group

def getGroup(lang):
    if lang in iso2group:
        return int(group2name[iso2group[lang]][0])
    return -1

# String distances from the AsjpDist-full program
# @author: rarakar
UNNORM = False
def levenshtein(a,b):
        m=[];la=len(a)+1;lb=len(b)+1
        for i in range(0,la):
                m.append([])
                for j in range(0,lb):m[i].append(0)
                m[i][0]=i
        for i in range(0,lb):m[0][i]=i
        for i in range(1,la):
                for j in range(1,lb):
                        s=m[i-1][j-1]
                        if (a[i-1]!=b[j-1]):s=s+1
                        m[i][j]=min(m[i][j-1]+1,m[i-1][j]+1,s)
        la=la-1;lb=lb-1
        if UNNORM:
                return float(m[la][lb])
        return float(m[la][lb])/float(max(la,lb))

aspj_conv = {}
for line in open('data/aspj_conv'):
    code, name = line.strip().split('\t')
    aspj_conv[code] = name

aspj_speakers = {}
aspj_data = {}
aspj_lines = open('data/lists.txt', encoding='ISO-8859-1').readlines()
for lineIdx, line in enumerate(aspj_lines):
    if line[0].isupper() and '{' in line:
        lang_name = line.split('{')[0]
        lang_code = aspj_lines[lineIdx+1].strip().split(' ')[-1]
        if len(lang_code) != 3:
            continue
        if lang_code in aspj_conv and lang_name == aspj_conv[lang_code]:
            data = []
            for i in range(100):
                if aspj_lines[lineIdx+2+i][0].isdigit():
                    line = aspj_lines[lineIdx+2+i].strip().replace(' //', '')
                    tok = line.split()
                    num = tok[0]
                    en = tok[1]
                    aspj_info = ' '.join(tok[2:])
                    data.append([num, aspj_info])
                else:
                    break
            aspj_data[lang_code] = data
            speakers = int(aspj_lines[lineIdx+1].strip().split()[3])
            if speakers >= 0:
                aspj_speakers[lang_code] = speakers


def get_aspj_speakers(lang):
    if lang in aspj_speakers:
        return aspj_speakers[lang]
    return 0



aes = {}
for line in open('data/aes.csv'):
    iso, status = line.strip().split(',')
    aes[iso] = status

def getAES(lang):
    if lang in aes:
        return aes[lang][0]
    return -1




geo = {}
for line in open('data/geo.csv'):
    iso, lat, long = line.strip().split(',')
    geo[iso] = [float(lat), float(long)]

mean_lat, mean_long = sum([g[0] for g in list(geo.values())])/len(geo), sum([g[1] for g in list(geo.values())])/len(geo)
def getGeo(lang):
    if lang in geo:
        return geo[lang]
    return mean_lat, mean_long



def getTrainDevTest(path):
    train = ''
    dev = ''
    test = ''
    for conlFile in os.listdir(path):
        if conlFile.endswith('conllu'):
            if 'train' in conlFile:
                train = path + '/' + conlFile
            if 'dev' in conlFile:
                dev = path + '/' + conlFile
            if 'test' in conlFile:
                test = path + '/' + conlFile
    return train, dev, test

def hasColumn(path, idx, threshold=.1):
    total = 0
    noWord = 0
    for line in open(path).readlines()[:5000]:
        if line[0] == '#' or len(line) < 2:
            continue
        tok = line.strip().split('\t')
        if tok[idx] == '_':
            noWord += 1
        total += 1
    return noWord/total < threshold

def load_json(path: str):
    import _jsonnet
    """
    Loads a jsonnet file through the json package and returns a dict.
    
    Parameters
    ----------
    path: str
        the path to the json(net) file to load
    """
    return json.loads(_jsonnet.evaluate_snippet("", '\n'.join(open(path).readlines())))

def getModel(name):
    modelDir = 'machamp/logs/'
    nameDir = modelDir + name + '/'
    if os.path.isdir(nameDir):
        for modelDir in reversed(os.listdir(nameDir)):
            modelPath = nameDir + modelDir + '/model.pt'
            if os.path.isfile(modelPath):
                return modelPath
    return ''


def extract_features(classifier: Literal['find_missing', 'find_value'], n_components:int = 10, dimension_reduction_method: str = 'pca', n: int = None):
    '''
    the main structure is that we get for each cell in the lang2vec matrix
    (language +feature) a gold value (in y), and a list of features describing
    this cell (in x). The features can be based on the language, on the feature,
    or on the text. We will then shuffle them, and do a k-fold prediction (to get
    an idea of performance). In the end we can use confidence thresholds or false
    negatives to get our "likely to be missing, but we have a gold label" cells.
    '''

    langs, vectors, vectors_knn = pickle.load(open('lang2vec.pickle', 'rb'))
    if n:
        langs, vectors, vectors_knn = langs[:n], vectors[:n], vectors_knn[:n]
    # Feature names can be split by '_' and used as features
    feature_names = l2v.get_features('eng', 'syntax_wals+phonology_wals', header=True)['CODE']
    # print(len(feature_names), len(langs))

    match classifier:
        case 'find_missing':
            # Create gold labels for missing values, note that it is of length 
            # #langs * #features
            # 0 indicates that the feature is missing, 1 indicates that it is present
            y = []
            names = []
            print('create y and names for find_missing classifier:')
            for vector, lang in tqdm(zip(vectors, langs), total=len(vectors)):
                gold_labels = [0 if val == -100 else 1 for val in vector]
                y.extend(gold_labels)
                for feature in feature_names:
                    names.append(lang + '|' + feature)
            print(f'-> {len(y)} datapoints extracted for the find_missing classifier')

        case 'find_value':
            # Create gold labels for present values, note that it is of length 
            # is less than #langs * #features
            # gold values includes 0, 1
            presIdxes = []
            y = []
            names = []
            print('create y and names for find_value classifier:')
            for langIdx, (vector, lang) in tqdm(enumerate(zip(vectors, langs)), total=len(vectors)):
                for featureIdx, (val, feature) in enumerate(zip(vector, feature_names)):
                    if val != -100:
                        y.append(val)
                        names.append(lang + '|' + feature)
                        presIdxes.append((langIdx, featureIdx))
            print(f'-> {len(y)} datapoints extracted for the find_value classifier')
        
        case _:
            print(f'classifier must be either "find_missing" or "find_value" NOT {classifier}')
            exit(1)       


    if n_components != 0:
        # load pylogency matrix 
        phyl_matrix_sparse = pickle.load(open('phyl-matrix-sparse.pickle', 'rb'))
        phyl_matrix = phyl_matrix_sparse.toarray()

        # reduce dimension of phylogency by PCA. There are other options like SVD, t-SNE, ... that may worth to try
        # Another idea for dimension reduction: look at this  plt.plot(np.sum(phyl_matrix, axis=0))
        match dimension_reduction_method:
            case 'pca':
                pca = PCA(n_components=n_components)  # Reduce to 100 dimensions
                phyl_matrix_pca = pca.fit_transform(phyl_matrix)
            case 'svd':
                raise
            case 't-sne':
                raise


    # Create features
    x = {'lang_id': [], 'feat_id': [], 'geo_lat': [], 'geo_long': [], 'lang_group': [], 'aes_status': [], 'wiki_size': [], 'num_speakers': [], 'lang_fam': [], 'scripts': [], 'feat_name': []}
    if n_components != 0:
        x.update({f'phylogency{i}':[] for i in range(n_components)})

    match classifier:
        case 'find_missing':
            print('create x(features) for the find_missing classifier:')
        case 'find_value':
            print('create x(features) for the find_value classifier:')

    for langIdx, lang in tqdm(enumerate(langs), total=len(langs)):
        if n_components != 0:
            phyl = phyl_matrix_pca[langIdx]
        geo = getGeo(lang)
        group = getGroup(lang)
        aes = getAES(lang)
        wiki_size = getWikiSize(lang)
        aspj_speakers = get_aspj_speakers(lang)
        fam = get_fam(lang)
        scripts = getScripts(lang)

        for featureIdx, feat_name in enumerate(feature_names):
            match classifier:
                case 'find_value':
                    if (langIdx, featureIdx) not in presIdxes:
                        continue
            # instanceIdx = langIdx * len(feature_names) + featureIdx
            # language identifier
            x['lang_id'].append(str(langIdx))

            # Add feature location
            x['feat_id'].append(str(featureIdx))
            
            # latitude and longitude from glottolog
            x['geo_lat'].append(geo[0])
            x['geo_long'].append(geo[1])

            if n_components != 0:
                # Pylogency feature from lang2vec
                for i in range(n_components):
                    x[f'phylogency{i}'].append(phyl[i])

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

    print('generating features done!')

    def underline_tok(line):
        return line.split('_')

    # https://scikit-learn.org/stable/modules/compose.html#columntransformer-for-heterogeneous-data
    fam_vectorizer = CountVectorizer(binary=True, tokenizer=underline_tok, ngram_range=(1, 1), analyzer='word')
    script_vectorizer = CountVectorizer(binary=True, tokenizer=underline_tok, ngram_range=(1, 1), analyzer='word')
    featname_vectorizer = CountVectorizer(binary=True, tokenizer=underline_tok, ngram_range=(1, 1), analyzer='word')

    phylogency_trans = [(f'phylogency{i}', MinMaxScaler(), [f'phylogency{i}']) for i in range(n_components)]

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

    # for i in range(10):
    #     print(names[i], y[i], x[i][:10])

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

    # with open(f'feats_{classifier}.pickle', 'wb') as f:
    #     pickle.dump([train_x, train_y, train_names, dev_x, dev_y, dev_names, all_feat_names], f)

    with open(f'feats-full_{classifier}.pickle', 'wb') as f:
        pickle.dump([x, y, names, all_feat_names], f)
