import os, pickle, datetime, argparse, string
import numpy as np
import pandas as pd
# BLOSUM62
Max_blosum = np.array([4, 5, 6, 6, 9, 5, 5, 6, 8, 4, 4, 5, 5, 6, 7, 4, 5, 11, 7, 4])
Min_blosum = np.array([-3, -3, -4, -4, -4, -3, -4, -4, -3, -4, -4, -3, -3, -4, -4, -3, -2, -4, -3, -3])
error_code_dic = {"PDB not exist": 1, "chain not exist": 2, "PDB_seq & dismap_seq mismatch": 3, "DSSP too long": 4, "Fail to pad DSSP": 5}


def BLOSUM_embedding(ID, seq, data_path):
    seq_embedding = []
    with open("blosum_dict.pkl", "rb") as f:
        blosum_dict = pickle.load(f)
    for aa in seq:
        seq_embedding.append(blosum_dict[aa])
    seq_embedding = (np.array(seq_embedding) - Min_blosum) / (Max_blosum - Min_blosum)
    np.save(data_path + "blosum/" + ID, seq_embedding)


def feature_extraction(PDBID, pdb_file, chain, mode, data_path):
    ID = PDBID + chain

    PDB_seq, error_code = get_PDB(PDBID, pdb_file, chain, data_path)
    if error_code != 0:
        return error_code

    with open(data_path + ID + ".fa", "w") as f:
        f.write(">" + ID + "\n" + PDB_seq)

    if mode == "fast":
        BLOSUM_embedding(ID, PDB_seq, data_path)
    else:
        MSA(ID, data_path)

    error_code = get_dssp(ID, PDB_seq, data_path)
    if error_code != 0:
        return error_code

    error_code = get_distance_map(ID, PDB_seq, data_path)
    if error_code != 0:
        return error_code

    return 0