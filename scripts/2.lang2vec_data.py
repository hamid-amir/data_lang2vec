import lang2vec.lang2vec as l2v
import myutils
import os

langs = {}
for lang_file in os.listdir('data/MILTALE-CLEAN/'):
    lang_code = lang_file.split('.')[0]
    langs[lang_code] = lang_file


found = 0
data = []
data_knn = []
data_langs = []
for lang in l2v.LANGUAGES:
    if lang in myutils.iso639_conv:
        lang = myutils.iso639_conv[lang]
    if lang not in langs:
        continue
    print(lang)
    # or should we use union/avg
    langvec = l2v.get_features(lang, 'syntax_wals+phonology_wals')[lang]
    langvec_knn = l2v.get_features(lang, 'syntax_knn+phonology_knn')[lang]
    langvec = [-100.0 if x=='--' else x for x in langvec]
    data_langs.append(lang)
    data.append(langvec)
    data_knn.append(langvec_knn)
print(len(data))
import pickle
with open('data.pickle', 'wb') as f:
    pickle.dump([data_langs, data, data_knn], f)



