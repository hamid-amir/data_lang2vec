import os 
import statistics


datadir = 'data/MILTALE-CLEAN/'
langs = {}
for langfile in os.listdir(datadir):
    if langfile.endswith('cleaned'):
        lang = langfile.split('.')[0]
        if lang in langs:
            langs[lang].append(langfile)
        else:
            langs[lang] = [langfile]

def read_pos(path):
    data = []
    cur_sent = []
    for line in open(path):
        if len(line) < 2:
            data.append(cur_sent)
            cur_sent = []
        else:
            cur_sent.append(line.strip().split('\t'))
    return data

labels = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']


def distribution(data):
    pos_counts = {label: 0 for label in labels}
    for sent in data:
        for word_info in sent:
            if len(word_info) == 2:
                pos_counts[word_info[1]] += 1
    return pos_counts

for lang in langs:
    all_data = []
    for pos_file in langs[lang]:
        all_data.extend(read_pos(datadir + pos_file))
    distr = distribution(all_data)
    freqs = distr.values()
    # https://en.wikipedia.org/wiki/Skewness
    mean = sum(freqs)/len(freqs)
    stdev = statistics.stdev(freqs)
    print(lang, int(mean), int(stdev))



