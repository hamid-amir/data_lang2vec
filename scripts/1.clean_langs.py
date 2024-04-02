import myutils
import os

mapping = {'est': 'ekk', 'zho': 'cmn', 'grn':'gug', 'toki': 'tok', 'nep': 'npi', 'lav':'lvs', 'ara': 'arb', 'ori':'ory', 'msa': 'zlm', 'kom': 'kpv'}
mapping.update(myutils.iso639_conv)


for lang_file in os.listdir('data/MILTALE-CLEAN/'):
    lang_code = lang_file.strip().split('.')[0]
    if lang_code in mapping:
        lang_code = mapping[lang_code]
    #print(myutils.iso2lang(lang_code), lang_file)
    if lang_code not in myutils.iso639:
        print(lang_code)

