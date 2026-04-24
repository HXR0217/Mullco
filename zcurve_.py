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



def z_curve_48bit_sequences(sequences):
    NN = 'ACGT'
    z=[]
    encodings = []
    header = ['SampleName', 'label']
    for base in NN:
        for base1 in NN:
            for elem in ['x', 'y', 'z']:
                header.append('%s%s.%s' % (base, base1, elem))
    encodings.append(header)

    for sequence in sequences:
        code = []
        pos_dict = {}
        for i in range(len(sequence) - 2):
            if sequence[i: i + 3] in pos_dict:
                pos_dict[sequence[i: i + 3]] += 1
            else:
                pos_dict[sequence[i: i + 3]] = 1

        for base in NN:
            for base1 in NN:
                code += [
                    (pos_dict.get('%s%sA' % (base, base1), 0) + pos_dict.get('%s%sG' % (base, base1), 0) - pos_dict.get(
                        '%s%sC' % (base, base1), 0) - pos_dict.get('%s%sT' % (base, base1), 0)) / (len(sequence) - 2),
                    # x
                    (pos_dict.get('%s%sA' % (base, base1), 0) + pos_dict.get('%s%sC' % (base, base1), 0) - pos_dict.get(
                        '%s%sG' % (base, base1), 0) - pos_dict.get('%s%sT' % (base, base1), 0)) / (len(sequence) - 2),
                    # y
                    (pos_dict.get('%s%sA' % (base, base1), 0) + pos_dict.get('%s%sT' % (base, base1), 0) - pos_dict.get(
                        '%s%sG' % (base, base1), 0) - pos_dict.get('%s%sC' % (base, base1), 0)) / (len(sequence) - 2)
                    # z
                ]
        z.append(code)

    return np.array(z)


def z_curve_144bit_sequences(sequences):
    NN = 'ACGT'
    z=[]
    encodings = []
    header = ['SampleName', 'label']
    for base in NN:
        for base1 in NN:
            for pos in range(1, 4):
                for elem in ['x', 'y', 'z']:
                    header.append('Pos_%s_%s%s.%s' % (pos, base, base1, elem))
    encodings.append(header)

    for sequence in sequences:
        code = []
        pos1_dict = {}
        pos2_dict = {}
        pos3_dict = {}
        for i in range(len(sequence) - 2):
            if (i + 1) % 3 == 1:
                if sequence[i: i + 3] in pos1_dict:
                    pos1_dict[sequence[i: i + 3]] += 1
                else:
                    pos1_dict[sequence[i: i + 3]] = 1
            elif (i + 1) % 3 == 2:
                if sequence[i: i + 3] in pos2_dict:
                    pos2_dict[sequence[i: i + 3]] += 1
                else:
                    pos2_dict[sequence[i: i + 3]] = 1
            elif (i + 1) % 3 == 0:
                if sequence[i: i + 3] in pos3_dict:
                    pos3_dict[sequence[i: i + 3]] += 1
                else:
                    pos3_dict[sequence[i: i + 3]] = 1

        for base in NN:
            for base1 in NN:
                code += [
                    (pos1_dict.get('%s%sA' % (base, base1), 0) + pos1_dict.get('%s%sG' % (base, base1),
                                                                               0) - pos1_dict.get(
                        '%s%sC' % (base, base1), 0) - pos1_dict.get('%s%sT' % (base, base1), 0)) / (len(sequence) - 2),
                    # x
                    (pos1_dict.get('%s%sA' % (base, base1), 0) + pos1_dict.get('%s%sC' % (base, base1),
                                                                               0) - pos1_dict.get(
                        '%s%sG' % (base, base1), 0) - pos1_dict.get('%s%sT' % (base, base1), 0)) / (len(sequence) - 2),
                    # y
                    (pos1_dict.get('%s%sA' % (base, base1), 0) + pos1_dict.get('%s%sT' % (base, base1),
                                                                               0) - pos1_dict.get(
                        '%s%sG' % (base, base1), 0) - pos1_dict.get('%s%sC' % (base, base1), 0)) / (len(sequence) - 2)
                    # z
                ]
                code += [
                    (pos2_dict.get('%s%sA' % (base, base1), 0) + pos2_dict.get('%s%sG' % (base, base1),
                                                                               0) - pos2_dict.get(
                        '%s%sC' % (base, base1), 0) - pos2_dict.get('%s%sT' % (base, base1), 0)) / (len(sequence) - 2),
                    # x
                    (pos2_dict.get('%s%sA' % (base, base1), 0) + pos2_dict.get('%s%sC' % (base, base1),
                                                                               0) - pos2_dict.get(
                        '%s%sG' % (base, base1), 0) - pos2_dict.get('%s%sT' % (base, base1), 0)) / (len(sequence) - 2),
                    # y
                    (pos2_dict.get('%s%sA' % (base, base1), 0) + pos2_dict.get('%s%sT' % (base, base1),
                                                                               0) - pos2_dict.get(
                        '%s%sG' % (base, base1), 0) - pos2_dict.get('%s%sC' % (base, base1), 0)) / (len(sequence) - 2)
                    # z
                ]
                code += [
                    (pos3_dict.get('%s%sA' % (base, base1), 0) + pos3_dict.get('%s%sG' % (base, base1),
                                                                               0) - pos3_dict.get(
                        '%s%sC' % (base, base1), 0) - pos3_dict.get('%s%sT' % (base, base1), 0)) / (len(sequence) - 2),
                    # x
                    (pos3_dict.get('%s%sA' % (base, base1), 0) + pos3_dict.get('%s%sC' % (base, base1),
                                                                               0) - pos3_dict.get(
                        '%s%sG' % (base, base1), 0) - pos3_dict.get('%s%sT' % (base, base1), 0)) / (len(sequence) - 2),
                    # y
                    (pos3_dict.get('%s%sA' % (base, base1), 0) + pos3_dict.get('%s%sT' % (base, base1),
                                                                               0) - pos3_dict.get(
                        '%s%sG' % (base, base1), 0) - pos3_dict.get('%s%sC' % (base, base1), 0)) / (len(sequence) - 2)
                    # z
                ]
        z.append(code)

    return np.array(z)
id_train_x = read_fasta('independent_seqs')
id_train_y = read_label('independent_labels')
x, y = extract_data(id_train_x, id_train_y )

z_x= np.array(z_curve_48bit_sequences(x))
z_y= y
df = pd.DataFrame(z_x)
df.to_csv('test_x_z_48bit.csv', index=False, header=False, sep=" ")
print(z_x.shape)
df = pd.DataFrame(z_y)
df.to_csv('test_y_z_48bit.csv', index=False, header=False, sep=" ")

z_x= np.array(z_curve_144bit_sequences(x))
z_y= y
df = pd.DataFrame(z_x)
df.to_csv('test_x_z_144bit.csv', index=False, header=False, sep=" ")
print(z_x.shape)
df = pd.DataFrame(z_y)
df.to_csv('test_y_z_144bit.csv', index=False, header=False, sep=" ")
