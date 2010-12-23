#!/usr/bin/env python2.6
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#from __future__ import unicode_literals


def _parse_paramh():
    # use this function to generate paramdic from param.h and
    # paramprofit.h in sextractor source

    path_param_h = '/usr/local/src/sextractor/sextractor-2.8.6/src/param.h'
    path_paramprofit_h = '/usr/local/src/sextractor/sextractor-2.8.6/src/paramprofit.h'
    
    with open(path_param_h) as f:
        s1 = ''.join(f.readlines()[21:])

    with open(path_paramprofit_h) as f:
        s2 = ''.join(f.readlines()[16:])

    ss = s1 + s2

    notparsed = []
    items = []
    i2 = 0
    while 1:
        ss = ss[i2:]
        i1 = ss.find('{')
        if i1 < 0:
            break
        i2 = ss.find('}') + 2
        if i2 < 0:
            break
        substr = ss[i1+1:i2-2]

        ts = substr.split('\n')

        tss = ts[0].split(',')
        name = (tss[0].strip())[1:-1]
        desc = ((''.join(tss[1:])).strip())[1:-1]

        ts = [t.strip() for t in (''.join(ts[1:])).split(',')]

        if not (len(ts) == 5 or len(ts) == 7 or len(ts) == 9):
            # not a valid parameter info
            notparsed.append(substr)
            continue

        # c type to fits type
        ctype = ts[2]
        fitstypes = {'T_BYTE': 'B',
                     'T_SHORT': 'I',
                     'T_LONG': 'J',
                     'T_FLOAT': 'E',
                     'T_DOUBLE': 'D'}
        if ctype not in fitstypes:
            raise RuntimeError('unknown c type: %s' % ctype)
        fitstype = fitstypes[ctype]

        # c format to fits format
        cformat = ts[3][1:-1]
        fitsformat = ''.join([cformat[-1].upper(),
                              cformat[1:-1]])

        it = {'name': name,
              'desc': desc,
              'cdisp': ts[1],
              'ctype': ctype,
              'cfmt': ts[3][1:-1],
              'unit': ts[4][1:-1],
              'type': fitstype,
              'fmt': fitsformat}
        items.append(it)

    print('# Generated by mkparamdic.py of tsastro.sextractor.')
    print('# Output parameters are compatible with SExtractor 2.8.6.')
    print('# Copy and paste the dictionary to sexcat2fits.')
    print('#')
    print('# Not parsed:')
    for o in notparsed:
        print('#  %s' % o)
    print('paramdic = {')
    for o in items:
        name = o['name']
        for key in ['ctype', 'cdisp', 'name']:
            o.pop(key)
        
        print("    '%s': %s," % (name, str(o)))
    print('}')
    

if __name__ == '__main__':
    _parse_paramh()
    exit(0)
