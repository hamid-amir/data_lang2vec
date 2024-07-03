import os

data_dir = 'data/MILTALE-CLEAN/'
for datafile in os.listdir(data_dir):
    if datafile.endswith('.pos'):
        outPath = data_dir + datafile + '.cleaned'
        if os.path.isfile(outPath):
            continue
        print(data_dir + datafile)
        data = open(data_dir + datafile).readlines()
        if len(data) == 0:
            print('NODATA')
            continue
        if len(data[0].split('\t')) == 10:
            outFile = open(outPath, 'w')
            for line in data:
                split = line.split('\t')
                if len(split) > 2:
                    outFile.write(split[1] + '\t' + split[3] + '\n')
                else:
                    outFile.write('\n')
            outFile.close()

