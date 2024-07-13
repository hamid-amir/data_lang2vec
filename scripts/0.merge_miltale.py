import myutils
import os

mil_dir = 'data_miltale/MILTALE/'
train_files = os.listdir(mil_dir + 'train')
dev_files = os.listdir(mil_dir + 'devtest')
test_files = os.listdir(mil_dir + 'test')

cleaned_dir = 'data_miltale/MILTALE-CLEAN/'
if not os.path.isdir(cleaned_dir):
    os.mkdir(cleaned_dir)

mapping = {'est': 'ekk', 'zho': 'cmn', 'grn':'gug', 'toki': 'tok', 'nep': 'npi', 'lav':'lvs', 'ara': 'arb', 'ori':'ory', 'msa': 'zlm', 'kom': 'kpv'}
mapping.update(myutils.iso639_conv)


for trainFile in train_files:
    trainPath = mil_dir + 'train/' + trainFile
    if os.path.getsize(trainPath) == 0:
        continue
    devPath = ''
    devFile = trainFile.replace('-train.utf8', '-devtest.utf8')
    if devFile in dev_files:
        devPath = mil_dir + 'devtest/' + devFile
        if os.path.getsize(devPath) == 0:
            devPath = ''
    testFile = trainFile.replace('-train.utf8', '-test.utf8')
    if testFile in test_files:
        testPath = mil_dir + 'test/' + testFile
        if os.path.getsize(testPath) == 0:
            testPath = ''

    lang_code = trainFile.split('/')[-1].split('.')[0].split('_')[0]
    iso_code = myutils.code2iso(lang_code)
    if iso_code:
        if os.name == 'nt': # if operating system is windows.
            cmd = 'type ' + trainPath + ' ' + devPath + ' ' + testPath + ' > ' + cleaned_dir + lang_code + '.' + trainFile.replace('-train','')
            cmd = cmd.replace('/', '\\')
        else:
            cmd = 'cat ' + trainPath + ' ' + devPath + ' ' + testPath + ' > ' + cleaned_dir + lang_code + '.' + trainFile.replace('-train','')
            cmd = cmd.replace('(', '\\(').replace(')', '\\)').replace('\'', '\\\'')
        print(cmd)
        os.system(cmd)

