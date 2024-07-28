import os
import pickle
import os
import sys
sys.path.append(os.getcwd())
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from typing import Dict, List, Literal




def extract_from_pos(pos_tags_path: str, 
                     vectorizer: Literal['countVectorizer', 'TfidfVectorizer'] = 'countVectorizer',
                     filter_threshold: int = None) -> pickle:
    def extract(input_file, output_file):
        if os.name == 'nt':
            with open(input_file, 'r', encoding='latin1') as infile, open(output_file, 'w', encoding='latin1') as outfile:
                for line in infile:
                    parts = line.strip().split('\t')
                    if len(parts) > 1:
                        outfile.write(parts[1] + '\n')
        else:
            with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
                for line in infile:
                    parts = line.strip().split('\t')
                    if len(parts) > 1:
                        outfile.write(parts[1] + '\n')

    if filter_threshold:
        pos_scores_files = []
        with open('9.posScores.out', 'r') as file:
            for line in file:
                pos_scores_files.append(line.strip())
        good_pos_files = [file.split('/')[-1] for file in pos_scores_files if float(file.split()[0]) > int(filter_threshold)]
        output_path = os.path.join(os.path.dirname(pos_tags_path), "extracted_pos_filtered")
    else:
        output_path = os.path.join(os.path.dirname(pos_tags_path), "extracted_pos")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for lang_file in os.listdir(pos_tags_path):
        if filter_threshold:
            if lang_file not in os.listdir(output_path) and lang_file[:-8] in good_pos_files:
                extract(os.path.join(pos_tags_path, lang_file), os.path.join(output_path, lang_file))
        else:
            if lang_file not in os.listdir(output_path):
                extract(os.path.join(pos_tags_path, lang_file), os.path.join(output_path, lang_file))

    langs = [pos_file.split('.')[0] for pos_file in os.listdir(output_path)]
    paths = [os.path.join(output_path, pos_file) for pos_file in os.listdir(output_path)]
    # langs = [lang_file.split('.')[0] for lang_file in os.listdir(pos_tags_path)]
    # paths = [os.path.join(pos_tags_path, pos_file) for pos_file in os.listdir(pos_tags_path)]

    # def tok(line):
    #     parts = line.strip().split('\t')
    #     if len(parts) > 1:
    #         return parts[1]

    if vectorizer == 'countVectorizer':
        if os.name == 'nt':
            vec = CountVectorizer(input='filename', analyzer='word', ngram_range=(3,5), max_features=2000, encoding='latin1')
        else:
            vec = CountVectorizer(input='filename', analyzer='word', ngram_range=(3,5), max_features=2000)
        # vec = CountVectorizer(input='filename', tokenizer=tok, analyzer='word', ngram_range=(2,4))
    elif vectorizer == 'TfidfVectorizer':
        if os.name == 'nt':
            vec = TfidfVectorizer(input='filename', analyzer='word', ngram_range=(3,5), max_features=2000, encoding='latin1')
        else:
            vec = TfidfVectorizer(input='filename', analyzer='word', ngram_range=(3,5), max_features=2000)

    X = vec.fit_transform(paths)

    with open('miltale_extracted_feats.pickle', 'wb') as f:
        pickle.dump([langs, X], f)



def extract_from_raw(miltale_path: str,
                     vectorizer: Literal['countVectorizer', 'TfidfVectorizer'] = 'countVectorizer') -> pickle:
    data_paths = {}
    for lang_file in os.listdir(miltale_path):
        lang_code = lang_file.split('.')[0]
        if lang_code not in data_paths.keys():
            data_paths[lang_code] = miltale_path + lang_file

    langs = list(data_paths.keys())
    paths = data_paths.values()

    if vectorizer == 'countVectorizer':
        vec = CountVectorizer(input='filename', analyzer='char', decode_error='ignore', ngram_range=(2,2), max_features=2000)
    elif vectorizer == 'TfidfVectorizer':
        vec = TfidfVectorizer(input='filename', analyzer='char', decode_error='ignore', ngram_range=(2,2), max_features=2000)

    X = vec.fit_transform(paths)

    with open('miltale_extracted_feats.pickle', 'wb') as f:
        pickle.dump([langs, X], f)




if __name__ == '__main__':
    if sys.argv[1] == 'raw_text':
        extract_from_raw(miltale_path='data_miltale/MILTALE-CLEAN/', 
                         vectorizer='countVectorizer')
    elif sys.argv[1] == 'pos_tags':
            extract_from_pos(pos_tags_path='data_miltale/pos_tags/MILTALE-CLEAN', 
                             vectorizer='countVectorizer')
    elif sys.argv[1] == 'pos_tags_filtered':
            extract_from_pos(pos_tags_path='data_miltale/pos_tags/MILTALE-CLEAN', 
                             vectorizer='countVectorizer', 
                             filter_threshold=sys.argv[2])



# def extract_script_based(miltale_path:str) -> Dict[str,List[str]]:
#     script_based = {}
#     for lang_file in os.listdir(miltale_path):
#         lang_code = lang_file.split('.')[0]
#         script = myutils.getScripts(lang_code) # outputs a set
#         if script:
#             script = script.pop()
#             if script not in script_based.keys():
#                 script_based[script] = [miltale_path + lang_file]
#             else:
#                 script_based[script].append(miltale_path + lang_file)
#     return script_based

# script_based = extract_script_based('data_miltale/MILTALE-CLEAN/')
