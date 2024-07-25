import myutils
import os
import json

miltale_dir = 'data/MILTALE-CLEAN/'
lm = 'microsoft/infoxlm-large'

diverse = False
smoothing = .5
name = 'UD2.14-pos' + lm.replace('/', '_') + '.' + str(diverse) + '.' + str(smoothing) + '.all'
model_path = myutils.getModel(name).replace('machamp/', '')

files = []
for lang_file in os.listdir(miltale_dir):
    if len(files) > 100:
        cmd = 'python3 predict.py ' + model_path + ' '
        cmd += ' '.join(files)
        cmd += ' --dataset UD_English-EWT --raw_text --topn 17'
        print(cmd)
        files = []
        
    if lang_file.endswith('utf8'):
        lang_file = '../' + miltale_dir + lang_file
        if ')' in lang_file or ' ' in lang_file or '\'' in lang_file:
        #    continue
            lang_file = '"' + lang_file + '"'
        pos_file = lang_file + '.pos'
        if not os.path.isfile(pos_file[3:]):
            files.extend([lang_file, pos_file])
cmd = 'python3 predict.py ' + model_path + ' '
cmd += ' '.join(files)
cmd += ' --dataset UD_English-EWT --raw_text --topn 17 --max_sents 1000'
print(cmd)

