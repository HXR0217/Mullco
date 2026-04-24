


def kmer_record(k):  # 计算2mer到8mer

    _1mer = ['A', 'C', 'G', 'T']
    _2mer = {}
    _3mer = {}
    _4mer = {}
    _5mer = {}
    _6mer = {}
    _7mer = {}
    _8mer = {}

    for n1 in _1mer:
        for n2 in _1mer:
            _2mer[n1 + n2] = 0
            for n3 in _1mer:
                _3mer[n1 + n2 + n3] = 0
                for n4 in _1mer:
                    _4mer[n1 + n2 + n3 + n4] = 0
                    for n5 in _1mer:
                        _5mer[n1 + n2 + n3 + n4 + n5] = 0
                        for n6 in _1mer:
                            _6mer[n1 + n2 + n3 + n4 + n5 + n6] = 0
                            for n7 in _1mer:
                                _7mer[n1 + n2 + n3 + n4 + n5 +n6 + n7] = 0
                                for n8 in _1mer:
                                    _8mer[n1 + n2 + n3 + n4 + n5 +n6 + n7 + n8] = 0

    if k == 2:
        record = _2mer
    elif k == 3:
        record = _3mer
    elif k == 4:
        record = _4mer
    elif k == 5:
        record = _5mer
    elif k == 6:
        record = _6mer
    elif k == 7:
        record = _7mer
    elif k == 8:
        record = _8mer
    else:
        print('Please must be sure that k value is between 2 and 8.')

    return record


def kmer(seq,k):
    re = kmer_record(k)
    total = 0
    for i in range(0,len(seq)):
        if i > (len(seq)- k):
            break
        else:
            tmp = seq[i : (i + k)]
            if tmp in re:
                re[tmp] += 1
            total += 1

    kmer_fea = []
    for i in re.values():
        ave = i / total
        kmer_fea.append(ave)

    return kmer_fea




def GetBaseRatio(seq):
    '''calculate the A, C, G, T percentage of ORF'''
    A, C, G, T = 0, 0, 0, 0
    for i in range(0, len(seq)):
        if seq[i] == 'A':
            A += 1
        elif seq[i] == 'C':
            C += 1
        elif seq[i] == 'G':
            G += 1
        elif seq[i] == 'T':
            T += 1

    A = A / (len(seq))
    C = C / (len(seq))
    G = G / (len(seq))
    T = T / (len(seq))
    return [A, C, G, T]
