# download luismc swadesh list from https://www.luismc.com/home/static/pubs/lrec2016-swadesh-data.zip
# download panlex swadesh list from https://db.panlex.org/panlex_swadesh_20170103.zip
# then extract and place them in the data/ directory

import os


output_dir = 'data/swadesh_merged'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

panlex110_list = []
with open('data/panlex_swadesh/swadesh110/eng-000.txt', 'r') as file:
    for line in file:
        panlex110_list.append(line.strip())

panlex207_list = []
with open('data/panlex_swadesh/swadesh207/eng-000.txt', 'r') as file:
    for line in file:
        panlex207_list.append(line.strip())



def merge_files(luismc_file_path, panlex110_file_path, panlex207_file_path):
    file_path = next(path for path in (luismc_file_path, panlex110_file_path, panlex207_file_path) if path)
    file_name = file_path.split('/')[-1].split('.')[0].split('-')[0] + '.txt'
    

    if panlex110_file_path:
        lang_list_110 = []
        with open(panlex110_file_path, 'r') as file:
            for line in file:
                lang_list_110.append(line.strip())

        en2lang_110 = {}
        for en_word, lang_word in zip(panlex110_list, lang_list_110):
            if '\t' not in en_word and len(lang_word) > 1:
                en2lang_110[en_word] = lang_word
            elif '\t' in en_word and len(lang_word) > 1:
                for en_w in en_word.split('\t'):
                    en2lang_110[en_w] = lang_word

    if panlex207_file_path:
        lang_list_207 = []
        with open(panlex207_file_path, 'r') as file:
            for line in file:
                lang_list_207.append(line.strip())
        en2lang_207 = {}
        for en_word, lang_word in zip(panlex207_list, lang_list_207):
            if '\t' not in en_word and len(lang_word) > 1:
                en2lang_207[en_word] = lang_word
            elif '\t' in en_word and len(lang_word) > 1:
                for en_w in en_word.split('\t'):
                    en2lang_207[en_w] = lang_word

    with open(os.path.join(output_dir, file_name), 'w') as output_file:
        added_words = set()
        added_lines = set()

        if luismc_file_path is not None:
            with open(luismc_file_path, 'r') as input_file:
                for line in input_file:
                    stripped_line = line.strip()
                    en_word = stripped_line.split(':')[0]
                    new_line = stripped_line
                    if panlex110_file_path:
                        if en_word in en2lang_110:
                            new_word = en2lang_110[en_word]
                            if new_word not in new_line:
                                new_line = new_line + ' | ' + new_word
                                added_words.add(en_word)
                    if panlex207_file_path:
                        if en_word in en2lang_207:
                            new_word = en2lang_207[en_word]
                            if new_word not in new_line:
                                new_line += ' | ' + new_word
                                added_words.add(en_word)
                    if new_line + ' \n' not in added_lines:
                        added_lines.add(new_line + ' \n')
                        output_file.write(new_line + ' \n')

        if panlex110_file_path:
            for en_word, lang_word in zip(en2lang_110.keys(), en2lang_110.values()):
                if en_word not in added_words:
                    added_words.add(en_word)
                    if en_word + ': ' + lang_word + ' \n' not in added_lines:
                        added_lines.add(en_word + ': ' + lang_word + ' \n')
                        output_file.write(en_word + ': ' + lang_word + ' \n')

        if panlex207_file_path:
            for en_word, lang_word in zip(en2lang_207.keys(), en2lang_207.values()):
                if en_word not in added_words:
                    added_words.add(en_word)
                    if en_word + ': ' + lang_word + ' \n' not in added_lines:
                        added_lines.add(en_word + ': ' + lang_word + ' \n')
                        output_file.write(en_word + ': ' + lang_word + ' \n')



def get_file_path(base_dir, lang):
    file_name = f"{lang}-000.txt" if base_dir.endswith(('swadesh110', 'swadesh207')) else f"{lang}.txt"
    full_path = os.path.join(base_dir, file_name)
    return full_path if os.path.exists(full_path) else None

def merge_all_files():
    luismc_dir = 'data/swadesh data/data'
    panlex110_dir = 'data/panlex_swadesh/swadesh110'
    panlex207_dir = 'data/panlex_swadesh/swadesh207'

    all_langs = set(l.split('.')[0] for l in os.listdir(luismc_dir)) | set(l.split('-')[0] for l in os.listdir(panlex110_dir)) | set(l.split('-')[0] for l in os.listdir(panlex207_dir))

    for lang in all_langs:
        luismc_path = get_file_path(luismc_dir, lang)
        panlex110_path = get_file_path(panlex110_dir, lang)
        panlex207_path = get_file_path(panlex207_dir, lang)

        if any([luismc_path, panlex110_path, panlex207_path]):
            merge_files(luismc_path, panlex110_path, panlex207_path)


merge_all_files()
