import os
import json
import pickle
import lang2vec.lang2vec as l2v
import time
from tqdm import tqdm
import numpy as np
from typing import Literal, List, Dict
import random 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.decomposition import PCA, TruncatedSVD
from scipy.sparse import csr_matrix


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


def get_feature_vector(langs: list, feature_name: str):
    matrix = np.zeros((len(langs),len(l2v.get_features('eng', feature_name, header=True)['CODE'])))
    for i, lang in tqdm(enumerate(langs)):
        langvec = l2v.get_features(lang, feature_name, header=True)[lang]
        langvec = [-100.0 if x=='--' else x for x in langvec]
        matrix[i] = langvec

    return matrix


def extract_features(classifier: Literal['find_missing', 'find_value'], n_components:int = 10, miltate_n_components: int = 10, dimension_reduction_method: str = 'pca', n: int = None, remove_features: list = [], miltale_data: bool = False, job_number=0, use_filtered=True):
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
        indices = random.sample(range(len(langs)), n)
        langs = [langs[i] for i in indices]
        vectors = [vectors[i] for i in indices]
        vectors_knn = [vectors_knn[i] for i in indices]

    # Feature names can be split by '_' and used as features
    feature_names = l2v.get_features('eng', 'syntax_wals+phonology_wals', header=True)['CODE']
    # print(len(feature_names), len(langs))

    if classifier == 'find_missing':
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

    elif classifier == 'find_value':
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
        
    else:
        print(f'classifier must be either "find_missing" or "find_value" NOT {classifier}')
        exit(1)       

    if 'inventory_average' not in remove_features:
        inventory_matrix = get_feature_vector(langs=langs, feature_name='inventory_average')

    if 'phonology_average' not in remove_features:
        phonology_matrix = get_feature_vector(langs=langs, feature_name='phonology_average')

    if 'geo' not in remove_features:
        geo_matrix = get_feature_vector(langs=langs, feature_name='geo')

    if n_components != 0 and 'phylogency' not in remove_features:
        # load pylogency matrix 
        phyl_matrix_sparse = pickle.load(open('phyl-matrix-sparse.pickle', 'rb'))
        phyl_matrix = phyl_matrix_sparse.toarray()

        # reduce dimension of phylogency by PCA. There are other options like SVD, t-SNE, ... that may worth to try
        # Another idea for dimension reduction: look at this  plt.plot(np.sum(phyl_matrix, axis=0))
        if dimension_reduction_method == 'pca':
            pca = PCA(n_components=n_components)  # Reduce to 100 dimensions
            phyl_matrix_pca = pca.fit_transform(phyl_matrix)
        elif dimension_reduction_method == 'svd':
            raise
        elif dimension_reduction_method == 't-sne':
            raise
    
    if miltale_data:
        if use_filtered:
            miltale_langs, miltale_X_sparse = pickle.load(open('miltale_extracted_feats.pickle', 'rb'))
        else:
            miltale_langs, miltale_X_sparse = pickle.load(open('miltale_extracted_feats_unfiltered.pickle', 'rb'))
        miltale_X = miltale_X_sparse.toarray()
        miltale_pca = PCA(n_components=miltate_n_components)
        miltale_X = miltale_pca.fit_transform(miltale_X)
        # miltale_X dimension : (#langs , #cv_features)

    # Create features
    x = {'lang_id': [], 'feat_id': [], 'geo_lat': [], 'geo_long': [], 'lang_group': [], 'aes_status': [], 'wiki_size': [], 'num_speakers': [], 'lang_fam': [], 'scripts': [], 'feat_name': []}
    x = {key: [] for key in x.keys() if key not in remove_features}

    if 'geo' not in remove_features:
        x.update({f'geo{i}':[] for i in range(geo_matrix.shape[1])})
    if 'phonology_average' not in remove_features:
        x.update({f'phonology_average{i}':[] for i in range(phonology_matrix.shape[1])})
    if 'inventory_average' not in remove_features:
        x.update({f'inventory_average{i}':[] for i in range(inventory_matrix.shape[1])})
    if n_components != 0 and 'phylogency' not in remove_features:
        x.update({f'phylogency{i}':[] for i in range(n_components)})
    if miltale_data:
        x.update({f'miltale{i}':[] for i in range(len(miltale_X[0]))})

    if classifier == 'find_missing':
        print('create x(features) for the find_missing classifier:')
    elif classifier == 'find_value':
        print('create x(features) for the find_value classifier:')

    for langIdx, lang in tqdm(enumerate(langs), total=len(langs)):
        if 'geo' not in remove_features:
            geo_l2v = geo_matrix[langIdx]
        if 'inventory_average' not in remove_features:
            inventory = inventory_matrix[langIdx]
        if 'phonology_average' not in remove_features:
            phonology = phonology_matrix[langIdx]
        if n_components != 0 and 'phylogency' not in remove_features:
            phyl = phyl_matrix_pca[langIdx]
        if miltale_data:
            if lang not in miltale_langs:
                continue
            idx = miltale_langs.index(lang)
            miltale = miltale_X[idx]
        geo = getGeo(lang)
        group = getGroup(lang)
        aes = getAES(lang)
        wiki_size = getWikiSize(lang)
        aspj_speakers = get_aspj_speakers(lang)
        fam = get_fam(lang)
        scripts = getScripts(lang)

        for featureIdx, feat_name in enumerate(feature_names):
            if classifier == 'find_value':
                if (langIdx, featureIdx) not in presIdxes:
                    continue
            # instanceIdx = langIdx * len(feature_names) + featureIdx
            # language identifier
            if 'lang_id' not in remove_features:
                x['lang_id'].append(str(langIdx))

            # Add feature location
            if 'feat_id' not in remove_features:
                x['feat_id'].append(str(featureIdx))
            
            # latitude and longitude from glottolog
            if 'geo_lat' not in remove_features:
                x['geo_lat'].append(geo[0])
            if 'geo_long' not in remove_features:
                x['geo_long'].append(geo[1])

            if 'geo' not in remove_features:
                for i in range(geo_matrix.shape[1]):
                    x[f'geo{i}'].append(geo_l2v[i])

            if 'phonology_average' not in remove_features:
                for i in range(phonology_matrix.shape[1]):
                    x[f'phonology_average{i}'].append(phonology[i])

            if 'inventory_average' not in remove_features:
                for i in range(inventory_matrix.shape[1]):
                    x[f'inventory_average{i}'].append(inventory[i])

            if n_components != 0 and 'phylogency' not in remove_features:
                # Pylogency feature from lang2vec
                for i in range(n_components):
                    x[f'phylogency{i}'].append(phyl[i])

            if miltale_data:
                # miltale extracted feature
                for i in range(len(miltale_X[0])):
                    x[f'miltale{i}'].append(miltale[i])

            # Group from paper
            if 'lang_group' not in remove_features:
                x['lang_group'].append(group)

            # Group from glottolog
            if 'aes_status' not in remove_features:
                x['aes_status'].append(int(aes))
            
            # Wikipedia size
            if 'wiki_size' not in remove_features:
                x['wiki_size'].append(wiki_size)

            # Number of speakers
            if 'num_speakers' not in remove_features:
                x['num_speakers'].append(aspj_speakers)

            # Language family is just one string, could have been all hight levels
            # Language family
            if 'lang_fam' not in remove_features:
                x['lang_fam'].append(fam)

            # Scripts
            if 'scripts' not in remove_features:
                x['scripts'].append('_'.join(scripts))

            # feature name
            if 'feat_name' not in remove_features:
                x['feat_name'].append(feat_name)

    print('generating features done!')

    def underline_tok(line):
        return line.split('_')

    # https://scikit-learn.org/stable/modules/compose.html#columntransformer-for-heterogeneous-data
    fam_vectorizer = CountVectorizer(binary=True, tokenizer=underline_tok, ngram_range=(1, 1), analyzer='word')
    script_vectorizer = CountVectorizer(binary=True, tokenizer=underline_tok, ngram_range=(1, 1), analyzer='word')
    featname_vectorizer = CountVectorizer(binary=True, tokenizer=underline_tok, ngram_range=(1, 1), analyzer='word')

    column_trans_list = list()
    if 'lang_id' not in remove_features:
        column_trans_list.append(('lang_id', OneHotEncoder(dtype='int'), ['lang_id']))
    if 'feat_id' not in remove_features:
        column_trans_list.append(('feat_id', OneHotEncoder(dtype='int'), ['feat_id']))
    if 'geo_lat' not in remove_features:
        column_trans_list.append(('geo_lat', MinMaxScaler(), ['geo_lat']))
    if 'geo_long' not in remove_features:
        column_trans_list.append(('geo_long', MinMaxScaler(), ['geo_long']))
    if 'lang_fam' not in remove_features:
        column_trans_list.append(('lang_fam', fam_vectorizer, 'lang_fam'))
    if 'scripts' not in remove_features:
        column_trans_list.append(('scripts', script_vectorizer, 'scripts'))
    if 'feat_name' not in remove_features:
        column_trans_list.append(('feat_name', featname_vectorizer, 'feat_name'))
    if 'phylogency' not in remove_features:
        phylogency_trans = [(f'phylogency{i}', MinMaxScaler(), [f'phylogency{i}']) for i in range(n_components)]
        column_trans_list += phylogency_trans
    if 'inventory_average' not in remove_features:
        inventory_average_trans = [(f'inventory_average{i}', MinMaxScaler(), [f'inventory_average{i}']) for i in range(inventory_matrix.shape[1])]
        column_trans_list += inventory_average_trans
    if 'phonology_average' not in remove_features:
        phonology_average_trans = [(f'phonology_average{i}', MinMaxScaler(), [f'phonology_average{i}']) for i in range(phonology_matrix.shape[1])]
        column_trans_list += phonology_average_trans
    if 'geo' not in remove_features:
        geo_trans = [(f'geo{i}', MinMaxScaler(), [f'geo{i}']) for i in range(geo_matrix.shape[1])]
        column_trans_list += geo_trans
    if miltale_data:
        miltale_trans = [(f'miltale{i}', MinMaxScaler(), [f'miltale{i}']) for i in range(len(miltale_X[0]))]
        column_trans_list += miltale_trans

    column_trans = ColumnTransformer(
        column_trans_list,
        remainder='passthrough', verbose_feature_names_out=True)

    # why do we need pandas?
    import pandas as pd
    x = column_trans.fit_transform(pd.DataFrame(x))
    all_feat_names = column_trans.get_feature_names_out()
    if isinstance(x, np.ndarray):
        x_numpy = x
    else:
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
    # split1 = int(len(z) * .6)
    # split2 = int(len(z) * .8)
    # train_x = x[:split1]
    # dev_x = x[split1:split2]
    # train_y = y[:split1]
    # dev_y = y[split1:split2]
    # train_names = names[:split1]
    # dev_names = names[split1:split2]

    # with open(f'feats_{classifier}.pickle', 'wb') as f:
    #     pickle.dump([train_x, train_y, train_names, dev_x, dev_y, dev_names, all_feat_names], f)

    file_name = 'feats-full_find_missing.pickle' if classifier=='find_missing' else f'feats-full_find_value_mil_{job_number}.pickle'
    with open(file_name, 'wb') as f:
        pickle.dump([x, y, names, all_feat_names], f)



# syntax_wals+phonology_wals
target_features = [
    'S_DEMONSTRATIVE_WORD_BEFORE_NOUN', 'S_INDEFINITE_AFFIX', 'S_SEX_MARK', 'S_VOS', 'S_OBJECT_BEFORE_VERB',
    'S_OBJECT_AFTER_VERB', 'S_NEGATIVE_WORD_AFTER_OBJECT', 'S_NUMCLASS_MARK', 'P_UVULAR_STOPS', 'P_CLICKS',
    'S_CASE_SUFFIX', 'S_XVO', 'S_NEGATIVE_WORD_BEFORE_SUBJECT', 'P_UVULAR_CONTINUANTS', 'S_POLARQ_AFFIX',
    'S_CASE_PROCLITIC', 'P_IMPLOSIVES', 'S_POLARQ_WORD', 'S_OBLIQUE_AFTER_VERB', 'S_DEFINITE_AFFIX',
    'S_INDEFINITE_WORD', 'P_VELAR_NASAL', 'P_NASAL_VOWELS', 'S_RELATIVE_AROUND_NOUN', 'P_BILABIALS',
    'S_POSSESSIVE_HEADMARK', 'S_SUBJECT_BEFORE_OBJECT', 'S_TEND_SUFFIX', 'P_TONE', 'S_OSV', 'P_VELAR_NASAL_INITIAL',
    'S_GENDER_MARK', 'P_FRICATIVES', 'P_VOICED_PLOSIVES', 'S_DEMONSTRATIVE_PREFIX', 'S_SUBORDINATOR_WORD_BEFORE_CLAUSE',
    'P_COMPLEX_CODAS', 'S_CASE_PREFIX', 'P_EJECTIVES', 'P_CODAS', 'S_OBLIQUE_BEFORE_VERB', 'S_OBJECT_HEADMARK',
    'S_ADJECTIVE_AFTER_NOUN', 'S_DEMONSTRATIVE_SUFFIX', 'S_NEGATIVE_PREFIX', 'S_CASE_MARK', 'S_COMITATIVE_VS_INSTRUMENTAL_MARK',
    'S_NEGATIVE_WORD_AFTER_VERB', 'S_SOV', 'S_NEGATIVE_WORD_ADJACENT_AFTER_VERB', 'S_NUMERAL_AFTER_NOUN', 'S_ADJECTIVE_WITHOUT_NOUN',
    'S_PLURAL_PREFIX', 'S_DEGREE_WORD_AFTER_ADJECTIVE', 'S_POSSESSIVE_DEPMARK', 'S_ADPOSITION_AFTER_NOUN', 'S_SUBJECT_BEFORE_VERB',
    'S_PLURAL_WORD', 'P_NASALS', 'S_TEND_PREFIX', 'S_SUBORDINATOR_WORD_AFTER_CLAUSE', 'S_OVS', 'S_NOMINATIVE_VS_ACCUSATIVE_MARK',
    'S_DEGREE_WORD_BEFORE_ADJECTIVE', 'S_RELATIVE_AFTER_NOUN', 'S_SVO', 'S_OXV', 'P_LATERAL_L', 'P_LABIAL_VELARS', 'S_DEMONSTRATIVE_WORD_AFTER_NOUN',
    'P_LATERAL_OBSTRUENTS', 'S_VSO', 'S_NEGATIVE_WORD_ADJACENT_BEFORE_VERB', 'S_POSSESSOR_AFTER_NOUN', 'P_PHARYNGEALS', 'P_VOICED_FRICATIVES',
    'S_OBLIQUE_BEFORE_OBJECT', 'S_POLARQ_MARK_FINAL', 'S_PLURAL_SUFFIX', 'S_NEGATIVE_WORD_AFTER_SUBJECT', 'S_SUBJECT_AFTER_VERB',
    'S_PERFECTIVE_VS_IMPERFECTIVE_MARK', 'S_NEGATIVE_WORD_FINAL', 'S_TAM_PREFIX', 'S_VOX', 'S_POSSESSIVE_PREFIX', 'P_GLOTTALIZED_RESONANTS',
    'S_PROSUBJECT_CLITIC', 'P_TH', 'S_NEGATIVE_SUFFIX', 'S_DEFINITE_WORD', 'S_FUTURE_AFFIX', 'S_RELATIVE_BEFORE_NOUN', 'S_XOV',
    'S_ERGATIVE_VS_ABSOLUTIVE_MARK', 'S_NEGATIVE_AFFIX', 'P_UVULARS', 'S_NUMERAL_BEFORE_NOUN', 'S_NEGATIVE_WORD_BEFORE_VERB', 'S_OVX',
    'S_CASE_ENCLITIC', 'S_POLARQ_MARK_SECOND', 'S_ADJECTIVE_BEFORE_NOUN', 'S_PROSUBJECT_WORD', 'S_ADPOSITION_BEFORE_NOUN',
    'S_POLARQ_MARK_INITIAL', 'S_NEGATIVE_WORD_INITIAL', 'S_TEND_DEPMARK', 'P_COMPLEX_ONSETS', 'S_POSSESSOR_BEFORE_NOUN',
    'P_FRONT_ROUND_VOWELS', 'P_LATERALS', 'S_PAST_VS_PRESENT_MARK', 'S_SUBJECT_AFTER_OBJECT', 'S_TEND_HEADMARK', 'S_OBJECT_DEPMARK',
    'S_NEGATIVE_WORD', 'S_TAM_SUFFIX', 'S_NEGATIVE_WORD_BEFORE_OBJECT', 'S_ANY_REDUP', 'S_PROSUBJECT_AFFIX', 'S_OBLIQUE_AFTER_OBJECT',
    'S_POSSESSIVE_SUFFIX', 'P_VOICE', 'S_SUBORDINATOR_SUFFIX',
]