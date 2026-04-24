# 姓名：黄
# 日期：2022.04.16

import itertools
from datetime import datetime as dt
import os,sys,re
from Bio import SeqIO
import pickle as pkl
import pandas as pd
import numpy as np
from skmultilearn.ensemble import RakelO
from TrainingMetrics import evalu_kfold
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
def read_fasta(inputfile):
    if os.path.exists(inputfile) == False:
        print('Error: file " %s " does not exist.' % inputfile)
        sys.exit(1)
    with open(inputfile) as f:
        record = f.readlines()
    if re.search('>', record[0]) == None:
        print('Error: the input file " %s " must be fasta format!' % inputfile)
        sys.exit(1)

    data = {}

    for line in record:
        if line.startswith('>'):
            name = line.replace('>', '').split('\n')[0]
            data[name] = ''
        else:
            data[name] += line.replace('\n', '')


    return data



def get_label_matrix(label_split):
    label_position = {'Exosome':0,'Nucleus':1,'Nucleoplasm':2,'Chromatin':3,'Cytoplasm':4,'Nucleolus':5, 'Cytosol':6,'Membrane':7, 'Ribosome':8}
    mm = [0,0,0,0,0,0,0,0,0]
    for i in label_split:
        if i in label_position:
            mm[label_position[i]] = 1
    return mm



def read_label(inputfile):
    input = pd.read_csv(inputfile, sep='\t')
    dic = {}
    for A, B, C in zip(input['Gene_ID'], input['Refseq_ID'],input['Annotation_label']):
        CC = C.split('|')
        label_list = get_label_matrix(CC)
        dic[A+'|'+B] = label_list
    return dic


def extract_data(input_seqs, input_labels):
    x=[]
    y=[]
    for i in input_labels:
        seq = input_seqs[i]
        x.append(seq)
        ttmp = input_labels[i]
        y.append(ttmp)
    x = np.array(x)
    y = np.array(y)

    return x, y


myDiIndex = {
    'AA': 0, 'AC': 1, 'AG': 2, 'AT': 3,
    'CA': 4, 'CC': 5, 'CG': 6, 'CT': 7,
    'GA': 8, 'GC': 9, 'GG': 10, 'GT': 11,
    'TA': 12, 'TC': 13, 'TG': 14, 'TT': 15
}

baseSymbol = 'ACGT'


def get_kmer_frequency(sequence, kmer):
    myFrequency = {}
    for pep in [''.join(i) for i in list(itertools.product(baseSymbol, repeat=kmer))]:
        myFrequency[pep] = 0
    for i in range(len(sequence) - kmer + 1):
        myFrequency[sequence[i: i + kmer]] = myFrequency[sequence[i: i + kmer]] + 1
    for key in myFrequency:
        myFrequency[key] = myFrequency[key] / (len(sequence) - kmer + 1)
    return myFrequency


def correlationFunction(pepA, pepB, myIndex, myPropertyName, myPropertyValue):
    CC = 0
    for p in myPropertyName:
        CC = CC + (float(myPropertyValue[p][myIndex[pepA]]) - float(myPropertyValue[p][myIndex[pepB]])) ** 2
    return CC / len(myPropertyName)


def correlationFunction_type2(pepA, pepB, myIndex, myPropertyName, myPropertyValue):
    CC = 0
    for p in myPropertyName:
        CC = CC + float(myPropertyValue[p][myIndex[pepA]]) * float(myPropertyValue[p][myIndex[pepB]])
    return CC
def TriNcleotideComposition(sequence, base):
    trincleotides = [nn1 + nn2 + nn3 for nn1 in base for nn2 in base for nn3 in base]
    tnc_dict = {}
    for triN in trincleotides:
        tnc_dict[triN] = 0
    for i in range(len(sequence) - 2):
        tnc_dict[sequence[i:i + 3]] += 1
    for key in tnc_dict:
       tnc_dict[key] /= (len(sequence) - 2)
    return tnc_dict
def PseEIIP(fastas):
    base = 'ACGT'
    EIIP_dict = {
        'A': 0.1260,
        'C': 0.1340,
        'G': 0.0806,
        'T': 0.1335 }
    trincleotides = [nn1 + nn2 + nn3 for nn1 in base for nn2 in base for nn3 in base]
    EIIPxyz = {}
    for triN in trincleotides:
        EIIPxyz[triN] = EIIP_dict[triN[0]] + EIIP_dict[triN[1]] + EIIP_dict[triN[2]]
    encodings = []
    for sequence in fastas:
        code = []
        trincleotide_frequency = TriNcleotideComposition(sequence, base)
        code = code + [EIIPxyz[triN] * trincleotide_frequency[triN] for triN in trincleotides]
        encodings.append(code)
    return encodings

def get_theta_array(myIndex, myPropertyName, myPropertyValue, lamadaValue, sequence, kmer):
    thetaArray = []
    for tmpLamada in range(lamadaValue):
        theta = 0
        for i in range(len(sequence) - tmpLamada - kmer):
            theta = theta + correlationFunction(sequence[i:i + kmer],
                                                sequence[i + tmpLamada + 1: i + tmpLamada + 1 + kmer], myIndex,
                                                myPropertyName, myPropertyValue)
        thetaArray.append(theta / (len(sequence) - tmpLamada - kmer))
    return thetaArray


def get_theta_array_type2(myIndex, myPropertyName, myPropertyValue, lamadaValue, sequence, kmer):
    thetaArray = []
    for tmpLamada in range(lamadaValue):
        for p in myPropertyName:
            theta = 0
            for i in range(len(sequence) - tmpLamada - kmer):
                theta = theta + correlationFunction_type2(sequence[i:i + kmer],
                                                          sequence[i + tmpLamada + 1: i + tmpLamada + 1 + kmer],
                                                          myIndex,
                                                          [p], myPropertyValue)
            thetaArray.append(theta / (len(sequence) - tmpLamada - kmer))
    return thetaArray


def PCPseDNC(sequences):
    myPropertyName = ['Base stacking', 'Protein induced deformability', 'B-DNA twist', 'A-philicity', 'Propeller twist',
                      'Duplex stability:(freeenergy)', 'DNA denaturation', 'Bending stiffness', 'Protein DNA twist',
                      'Aida_BA_transition', 'Breslauer_dG', 'Breslauer_dH', 'Electron_interaction',
                      'Hartman_trans_free_energy', 'Helix-Coil_transition', 'Lisser_BZ_transition', 'Polar_interaction',
                      'SantaLucia_dG', 'SantaLucia_dS', 'Sarai_flexibility', 'Stability', 'Sugimoto_dG', 'Sugimoto_dH',
                      'Sugimoto_dS', 'Duplex tability(disruptenergy)', 'Stabilising energy of Z-DNA', 'Breslauer_dS',
                      'Ivanov_BA_transition', 'SantaLucia_dH', 'Stacking_energy', 'Watson-Crick_interaction',
                      'Dinucleotide GC Content', 'Twist', 'Tilt', 'Roll', 'Shift', 'Slide', 'Rise']
    dataFile = 'Phychepro.data'
    with open(dataFile, 'rb') as f:
        myPropertyValue = pkl.load(f)
    lamadaValue = 2  # 20
    weight = 0.1  # 0.9
    myIndex = myDiIndex
    PCPseDNC_feature = []
    for i in sequences:
        code = []
        dipeptideFrequency = get_kmer_frequency(i, 2)
        thetaArray = get_theta_array(myIndex, myPropertyName, myPropertyValue, lamadaValue, i, 2)
        for pair in sorted(myIndex.keys()):
            code.append(dipeptideFrequency[pair] / (1 + weight * sum(thetaArray)))
        for k in range(17, 16 + lamadaValue + 1):
            code.append((weight * thetaArray[k - 17]) / (1 + weight * sum(thetaArray)))
        PCPseDNC_feature.append(code)
    return PCPseDNC_feature


id_train_x = read_fasta('independent_seqs')
id_train_y = read_label('independent_labels')
x, y = extract_data(id_train_x, id_train_y )
PCP_x= np.array(PCPseDNC(x))
PCP_y= y
EIIP_x= np.array(PseEIIP(x))
EIIP_y= y


df = pd.DataFrame(PCP_x)
df.to_csv('test_x_PCPseDNC.csv', index=False, header=False, sep=" ")
print(PCP_x.shape)
df = pd.DataFrame(PCP_y)
df.to_csv('test_y_PCPseDNC.csv', index=False, header=False, sep=" ")
df = pd.DataFrame(EIIP_x)
df.to_csv('test_x_EIIP.csv', index=False, header=False, sep=" ")
print(PCP_x.shape)
df = pd.DataFrame(EIIP_y)
df.to_csv('test_y_EIIP.csv', index=False, header=False, sep=" ")
