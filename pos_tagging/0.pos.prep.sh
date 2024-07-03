
cd data 
wget https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-5502/ud-treebanks-v2.14.tgz
tar -zxvf ud-treebanks-v2.14.tgz
rm ud-treebanks-v2.14.tgz
cd ../

git clone https://github.com/machamp-nlp/machamp 
cd machamp
git reset --hard 9f5a6ce48fcebed353956f28de59d9d99098f073
python3 scripts/misc/cleanconl.py ../data/ud-treebanks-v2.14/*/*conllu
cd ../


