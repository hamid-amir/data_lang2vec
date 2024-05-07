seed = 8446

# Handling of iso639 codes
iso639 = {}
macros = set()
for line in open('data/iso-639-3.tab').readlines()[1:]:
    tok = line.strip().split('\t')
    if tok[4] == 'I':
        iso639[tok[0]] = tok[6]
    elif tok[4] == 'M':
        macros.add(tok[0])

iso639_conv = {}
for line in open('data/iso-639-3_Retirements.tab').readlines()[1:]:
    tok = line.strip().split('\t')
    prev = tok[0]
    new = tok[3]
    if new != '':
        iso639_conv[prev] = new

def code2iso(code):
    if code in iso639_conv:
        code = iso639_conv[code]

    if code in iso639: 
        return code
    else:
        return None
    #elif code in macros:
    #    return "MACRO"
    #elif '-' in code:
    #    return "DIALECT"
    #else:
    #    return None


# Handling of language families of glottolog
trees = []
from newick import loads
for line in open('data/tree_glottolog_newick.txt'):
    tree = loads(line.strip())
    trees.append(tree[0])

def getPath(langCode):
    found = None
    for tree in trees:
        if found != None:
            break
        for item in tree.walk():
            if '[' + langCode + ']' in item.name:
                found = item
    if found == None:
        return None
    curNode = found
    path = [curNode.name]
    while hasattr(curNode, 'ancestor') and curNode != None:
        curNode = curNode.ancestor
        if curNode == None:
            break
        path.append(curNode.name)
    return path

def get_fam(lang):
    found = None
    for tree in trees:
        if found != None:
            break
        for item in tree.walk():
            if '[' + lang + ']' in item.name:
                found = item
    if found == None:
        return None
    curNode = found
    return curNode.name.split('[')[1].split(']')[0]


def treeDist(path1, path2):
    # The percentage of trees of distance. This means that if you are in two
    # different trees, it will also be 2.0 If both languages are in the same
    # tree it is #overlapping edges/the total edges of the deepest language of
    # the two.
    if path1 == None or path2 == None:
        return None

    overlap_counter = 0
    found = False
    for item in path1:
        if item in path2:
            overlap_counter += 1
    if overlap_counter == 0:
        return 2.0
    overlap_tree1 = 1- overlap_counter/len(path1)
    overlap_tree2 = 1- overlap_counter/len(path2)
    return overlap_tree1 + overlap_tree2

# Reading wikipedia sizes. They use two character language codes, 
# so we also need the conversion
two2three = {}
threes = set()
lang2code = {}
for line in open('data/iso-639-3.tab').readlines()[1:]:
    tok = line.strip().split('\t')
    if tok[3] != '':
        two2three[tok[3]] = tok[0]
    threes.add(tok[0])
    lang2code[tok[6]] = tok[0]
wiki_sizes = []
for line in open('data/List_of_Wikipedias'):
    if line.startswith('<td><a href="https://en.wikipedia.org/wiki/') and '_language' in line:
        lang = line.strip().split('<')[-3].split('>')[-1]
    if line.startswith('<td><bdi lang='):
        lang_code = line.strip().split('"')[1].split('-')[0]
        if lang_code in two2three:
            wiki_sizes.append([two2three[lang_code], 0])
        elif lang_code in threes:
            wiki_sizes.append([lang_code, 0])
        elif lang in lang2code:
            wiki_sizes.append([lang2code[lang], 0])
    if 'rg/wiki/Special:Statistics" class="extiw" title=' in line:
        size = line.strip().split('>')[2].split('<')[0]
        wiki_sizes[-1][1] = int(size.replace(',', ''))
# convert to dict
wiki_sizes = {lang:size for lang, size in wiki_sizes}
def getWikiSize(lang):
    if lang in wiki_sizes:
        return wiki_sizes[lang]
    return 0

# Read glotscript data. Note that Brai (braille) is removed, because
# annotation is incomplete
glotscript_data = {}
for line in open('data/GlotScript.tsv').readlines()[1:]:
    tok = line.strip().lower().split('\t')
    if len(tok) > 1:
        glotscript_data[tok[0]] = set([x.strip() for x in tok[1].split(',') if x != ' brai'])
def getScripts(lang):
    if lang in glotscript_data:
        return glotscript_data[lang]
    else:
        return []

group2name = {'0': '0. The Left-Behinds', '1': '1. The Scraping-Bys', '2': '2. The Hopefuls', '3': '3. The Rising Stars', '4': '4. The Underdogs', '5': '5. The Winners'}
iso2group = {}
for line in open('data/lang2tax.txt.codes'):
    iso, group, name = line.strip().split(',') 
    iso2group[iso] = group

def getGroup(lang):
    if lang in iso2group:
        return int(group2name[iso2group[lang]][0])
    return -1

# String distances from the AsjpDist-full program
# @author: rarakar
UNNORM = False
def levenshtein(a,b):
        m=[];la=len(a)+1;lb=len(b)+1
        for i in range(0,la):
                m.append([])
                for j in range(0,lb):m[i].append(0)
                m[i][0]=i
        for i in range(0,lb):m[0][i]=i
        for i in range(1,la):
                for j in range(1,lb):
                        s=m[i-1][j-1]
                        if (a[i-1]!=b[j-1]):s=s+1
                        m[i][j]=min(m[i][j-1]+1,m[i-1][j]+1,s)
        la=la-1;lb=lb-1
        if UNNORM:
                return float(m[la][lb])
        return float(m[la][lb])/float(max(la,lb))

aspj_conv = {}
for line in open('data/aspj_conv'):
    code, name = line.strip().split('\t')
    aspj_conv[code] = name

aspj_speakers = {}
aspj_data = {}
aspj_lines = open('data/lists.txt', encoding='ISO-8859-1').readlines()
for lineIdx, line in enumerate(aspj_lines):
    if line[0].isupper() and '{' in line:
        lang_name = line.split('{')[0]
        lang_code = aspj_lines[lineIdx+1].strip().split(' ')[-1]
        if len(lang_code) != 3:
            continue
        if lang_code in aspj_conv and lang_name == aspj_conv[lang_code]:
            data = []
            for i in range(100):
                if aspj_lines[lineIdx+2+i][0].isdigit():
                    line = aspj_lines[lineIdx+2+i].strip().replace(' //', '')
                    tok = line.split()
                    num = tok[0]
                    en = tok[1]
                    aspj_info = ' '.join(tok[2:])
                    data.append([num, aspj_info])
                else:
                    break
            aspj_data[lang_code] = data
            speakers = int(aspj_lines[lineIdx+1].strip().split()[3])
            if speakers >= 0:
                aspj_speakers[lang_code] = speakers

def expand_list(data):
    new_data = [None] * 100
    for item in data:
        new_data[int(item[0])-1] = item[-1]
    return new_data

def get_aspj_speakers(lang):
    if lang in aspj_speakers:
        return aspj_speakers[lang]
    return 0

def get_aspj_dist(lang1, lang2):
    # https://aclanthology.org/W12-0212.pdf
    # "Adding typology to lexicostatistics"
    if lang1 not in aspj_data or lang2 not in aspj_data:
        return None
    data1 = expand_list(aspj_data[lang1])
    data2 = expand_list(aspj_data[lang2])
    dists = [[None]*100 for _ in range(100)]
    for idx1, item1 in enumerate(data1):
        for idx2, item2 in enumerate(data2):
            if None not in [item1, item2]:
                # TODO handle multiple items (,)
                all_dists = []
                for item1_alternative in item1.split(', '):
                    for item2_alternative in item2.split(', '):
                        all_dists.append(levenshtein(item1_alternative, item2_alternative))
                dists[idx1][idx2] = sum(all_dists)/len(all_dists)
    items_ldnd = []
    # get average over all non-matching pairs to normalize for chance
    all_dists = []
    for x in range(100):
        for y in range(100):
            if x != y and dists[x][y] != None:
                all_dists.append(dists[x][y])
    avg_dist = sum(all_dists)/len(all_dists)

    all_ldnd = []
    for item in range(100):
        if dists[item][item] != None:
            ldn = dists[item][item]
            ldnd = ldn/avg_dist 
            all_ldnd.append(ldnd)
    return sum(all_ldnd)/len(all_ldnd)

aes = {}
for line in open('data/aes.csv'):
    iso, status = line.strip().split(',')
    aes[iso] = status

def getAES(lang):
    if lang in aes:
        return aes[lang][0]
    return -1

