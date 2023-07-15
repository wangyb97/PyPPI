import re
import numpy as np


def splitFile(fileName):
    if not fileName.endswith(".fasta"):
        print("Please use a filename ending with fasta!")
        exit(1)
    seq_file = open(fileName, 'r')
    seqs = []
    seq_name = []
    for line in seq_file:
        if line.startswith('>'):
            seq_name.append(line[1:].strip())
        else:
            seqs.append(line.strip())
    seq_split = []

    for i in seqs:
        seq_split.append(str(' '.join([word for word in i])))
    sequences = [re.sub(r"[UZOB]", "X", sequence) for sequence in seq_split]
    return sequences


def getID(fileName):
    if not fileName.endswith(".fasta"):
        print("Please use a filename ending with fasta!")
        exit(1)
    seq_file = open(fileName, 'r')
    seq_name = []
    for line in seq_file:
        if line.startswith('>'):
            seq_name.append(line[1:].strip())
    return seq_name


def getLabel(label_npy=''):
    label = np.load(label_npy)
    # converted_label = [int(item) for item in label]
    # converted_data = [' '.join(str(item) for item in converted_label)]
    # converted_data = '[' + ' '.join(label) + ']'
    converted_data =np.array([int(x) for x in label])
    print(type(converted_data))
    return converted_data
