import os 
import json
import timeit 
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys


# check dir exists
def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def write_json(data_dict, data_id, save_dir, compl=False):
    data_item = data_dict[data_id]
    prot_seq = data_item['prot_seq']

    data = {
        "name": data_id,
        "sequences": [],
        "modelSeeds": [1],
        "dialect": "alphafold3",
        "version": 1
    }    
    
    data["sequences"].append(
        {
            "protein": {
                "id": 'A',
                "sequence": prot_seq
            }
        }
    )
    if compl:
        rna_seq = data_item['rna_seq']
        data["sequences"].append(
                {
                    "rna": {
                        "id": 'B',
                        "sequence": rna_seq
                    }
                }
            )
    
    save_path = os.path.join(save_dir, f'{data_id}.json')
    
    with open(save_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
        
    return save_path


def build_jsons(data_dict, save_dir, compl=False):
    check_dir(save_dir)
    data_ids = []
    json_paths = {}
    for pdb_id in data_dict.keys():
        json_path = write_json(data_dict, pdb_id, save_dir, compl)
        data_ids.append(pdb_id)
        json_paths[pdb_id] = json_path
    
    return data_ids, json_paths


def run_msa(data_id, json_path, alphafold_3_dir, msa_dir, cpu_num=8):
    path_for_check = os.path.join(msa_dir, f'{data_id.lower()}/{data_id.lower()}_data.json')
    if os.path.isfile(path_for_check):
        print(path_for_check)
        print(f'{data_id} has already been processed.')
    else:
        command = f'python {alphafold_3_dir}/run_alphafold.py \
                    --json_path {json_path} --output_dir {msa_dir} \
                    --jackhmmer_n_cpu {cpu_num} --nhmmer_n_cpu {cpu_num} --norun_inference'
                    
        start = timeit.default_timer()
        print(f'>Start working on {json_path}...')
        subprocess.run(command, shell=True, text=True)
        stop = timeit.default_timer()
        print(f'End {json_path}, using {stop - start} seconds.')
        

def process_batch_msa(data_dict, run_msa_calculation=run_msa, cpu_num=8, compl=False, alphafold_3_dir=None,
                      json_dir='AF3_pred/jsons', msa_dir='AF3_pred/structs'):
    check_dir(msa_dir)
    check_dir(json_dir)
    data_ids, json_paths = build_jsons(data_dict, json_dir, compl)
    with ProcessPoolExecutor(max_workers=cpu_num) as executor:
        future_to_seq = {
            executor.submit(run_msa_calculation, data_id, json_paths[data_id], alphafold_3_dir, msa_dir, cpu_num): data_id
            for data_id in data_ids
        }
        for future in as_completed(future_to_seq):
            data_id = future_to_seq[future]
            future.result() 
    return data_ids


def run_structure(data_id, alphafold_3_dir, struct_idr='AF3_pred/structs'):
    json_path = f'{struct_idr}/{data_id.lower()}/{data_id.lower()}_data.json'
    command = f'python {alphafold_3_dir}/run_alphafold.py \
                --json_path {json_path} --output_dir {struct_idr}/{data_id.lower()}/predicted --save_embeddings \
                --norun_data_pipeline'
    start = timeit.default_timer()
    print(f'>Start working on {json_path}...')
    subprocess.run(command, shell=True, text=True)
    stop = timeit.default_timer()
    print(f'End {json_path}, using {stop - start} seconds.')
    
    
def run_alphafold3_prediction(data_dict, cpu_num=8, compl=False, 
                              json_dir='AF3_pred/jsons',
                              alphafold_3_dir='./AlphaFold3', 
                              msa_dir='AF3_pred/structs'):
    data_ids = process_batch_msa(data_dict, cpu_num=cpu_num, compl=compl, json_dir=json_dir, alphafold_3_dir=alphafold_3_dir, msa_dir=msa_dir)
    for data_id in data_ids:
        run_structure(data_id, alphafold_3_dir, struct_idr=msa_dir)