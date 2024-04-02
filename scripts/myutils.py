iso639 = {}
macros = set()
for line in open('scripts/iso-639-3.tab').readlines()[1:]:
    tok = line.strip().split('\t')
    if tok[4] == 'I':
        iso639[tok[0]] = tok[6]
    elif tok[4] == 'M':
        macros.add(tok[0])

iso639_conv = {}
for line in open('scripts/iso-639-3_Retirements.tab').readlines()[1:]:
    tok = line.strip().split('\t')
    prev = tok[0]
    new = tok[3]
    if new != '':
        iso639_conv[prev] = new

def iso2lang(code):
    if code in iso639_conv:
        code = iso639_conv[code]

    if code in iso639: 
        return iso639[code]
    elif code in macros:
        return "MACRO"
    elif '-' in code:
        return "DIALECT"
    else:
        return None




