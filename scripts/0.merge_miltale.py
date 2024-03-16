import os

mil_dir = 'data/MILTALE/'
train_files = os.listdir(mil_dir + 'train')
dev_files = os.listdir(mil_dir + 'devtest')
test_files = os.listdir(mil_dir + 'test')

cleaned_dir = 'data/MILTALE-CLEAN/'
if not os.path.isdir(cleaned_dir):
    os.mkdir(cleaned_dir)

for trainFile in train_files:
    trainPath = mil_dir + 'train/' + trainFile
    devPath = ''
    devFile = trainFile.replace('-train.utf8', '-devtest.utf8')
    if devFile in dev_files:
        devPath = mil_dir + 'devtest/' + devFile
    testFile = trainFile.replace('-train.utf8', '-test.utf8')
    if testFile in test_files:
        testPath = mil_dir + 'test/' + testFile
    cmd = 'cat ' + trainPath + ' ' + devPath + ' ' + testPath + ' > ' + cleaned_dir + trainFile.replace('-train','')
    cmd = cmd.replace('(', '\\(').replace(')', '\\)').replace('\'', '\\\'')
    print(cmd)

