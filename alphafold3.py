import os
import pandas as pd
from data.run_alphafold3 import run_alphafold3_prediction
from parsing import parse_args
import random

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


if __name__ == '__main__':
    args = parse_args()
    
    if args.input.endswith('.fasta'):
        data_dict = fasta_from_file(args.input)
    elif args.input.endswith('.csv'):
        data_dict = read_csv(args.input)
    else:
        data_dict = fasta_from_dir(args.input)
        
    json_dir = f'{args.af3_dir}/jsons'
    msa_dir = f'{args.af3_dir}/structs'
    check_dir(json_dir)
    check_dir(msa_dir)
    
    print('Start running AlphaFold 3...')
    run_alphafold3_prediction(data_dict, cpu_num=args.cpu_num, 
                              compl=args.compl,
                              json_dir=json_dir,
                              msa_dir=msa_dir,
                              alphafold_3_dir=args.af3_exec_dir)