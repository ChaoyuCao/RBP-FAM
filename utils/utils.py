import pickle
import os
import numpy as np
import pandas as pd
import torch


# save and load pkl file
def pkl_saver(path, data):
    file = open(path, 'wb')
    pickle.dump(data, file)
    file.close()


# read pkl file
def pkl_loader(path):
    file = open(path, 'rb')
    data = pickle.load(file)
    return data


# check dir exists
def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
        
def fasta_from_dir(main_dir):
    filenames = os.listdir(main_dir)
    proteins = {}
    for filename in filenames:
        filepath = os.path.join(main_dir, filename)
        with open(filepath, 'r') as file:
            lines = file.readlines()
        prot_id = lines[0][1:-1]
        seq = lines[1]
        if seq.endswith('\n'):
            seq = seq[:-1]
        proteins[prot_id] = {'prot_seq': seq}
    return proteins


def fasta_from_file(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
    proteins = {}
    for i, line in enumerate(lines):
        if not line.startswith('>'):
            continue
        prot_id = line[1:-1]
        seq = lines[i + 1]
        if seq.endswith('\n'):
            seq = seq[:-1]
        proteins[prot_id] = {'prot_seq': seq}
    return proteins


def read_csv(filepath):
    df = pd.read_csv(filepath)
    df.columns = [str(col).strip() for col in df.columns]
    data_dict = {}
    for _, row in df.iterrows():
        data_dict[row['ID']] = {
            'prot_seq': row['Protein seq'],
            'rna_seq': row['RNA seq']
        }
    return data_dict


def lst_to_tensor(lst, dtype_=torch.float32):
    tensors = []
    for array in lst:
        if not isinstance(array, torch.Tensor):
            tensor = torch.tensor(array, dtype=dtype_)
        else:
            tensor = array
        tensors.append(tensor)
    return tensors