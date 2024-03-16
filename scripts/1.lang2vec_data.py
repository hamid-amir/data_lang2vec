
import lang2vec.lang2vec as l2v

lang = 'nld'
if lang in l2v.LANGUAGES:
    langvec = l2v.get_features(lang, 'syntax_knn+phonology_knn')[lang]

