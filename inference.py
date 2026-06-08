import torch
from utils.utils import *
from torch_geometric.loader import DataLoader
from data.run_alphafold3 import run_alphafold3_prediction
from data.prepare_data import DataGenerator
from parsing import parse_args
from modules.GNN import build_model
import random
import tqdm
import math
import csv


def delta_G_to_Kd(delta_G, temperature=298, unit='nM'):
    R_kcal = 0.001987
    RT = R_kcal * temperature
    Kd_M = math.exp(delta_G / RT)
    units = {
        'M': 1, 'mM': 1e-3, 'uM': 1e-6, 'nM': 1e-9,
        'pM': 1e-12, 'fM': 1e-15
    }
    factor = units.get(unit.lower(), 1e-9)
    return Kd_M / factor


if __name__ == '__main__':
    args = parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    
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
    check_dir(args.save_dir)
    
    print('Predicting...')
    preds = []
    binaries = []
    attns = {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}
    for fold_idx in range(5):
        print('=' * 10 + f'Fold {fold_idx}' + '=' * 10)
        model = build_model(args)
        model_path = f'./weights/{args.task}/fold_{fold_idx}.pth'
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.to(args.device)
        model.eval()
        
        dataset = DataGenerator(args, data_dict)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        fold_preds = []
        fold_binaries = []
        attns_fold = []
        all_ids = []

        for idx, batch in enumerate(tqdm.tqdm(dataloader)):
            data_id, X, h_V, edge_index, batch_id, mask = batch.name, batch.X, batch.node_feat, batch.edge_index, batch.batch, batch.mask
            y_pred = model(X, h_V, edge_index, batch_id, mask)
            if args.task == 'binding_site':
                y_pred = torch.softmax(y_pred, dim=-1)
                fold_binaries.append(torch.argmax(y_pred, dim=-1)[:, None].float())
            fold_preds.append(y_pred[:, -1])
            
            all_ids += data_id
            if args.task == 'binding_site':
                attn_weights = model.Graph_encoder.layers[-1].saving_attn
            else:
                attn_weights = model.pooling_layer.saving_attn
            attns_fold.append(attn_weights)

        fold_preds = torch.cat(fold_preds, dim=0)
        fold_preds = fold_preds.unsqueeze(dim=-1)
        preds.append(fold_preds)
        if len(fold_binaries) > 0:
            fold_binaries = torch.cat(fold_binaries, dim=0)
            binaries.append(fold_binaries)

        for id_, attn in zip(all_ids, attns_fold):
            attns[fold_idx + 1][id_] = attn

    preds = torch.cat(preds, dim=-1)
    mean_preds = torch.mean(preds, dim=-1).detach().cpu().numpy().tolist()
    if len(binaries) > 0:
        binaries = torch.cat(binaries, dim=-1)
        binaries = torch.mean(binaries, dim=-1).detach().cpu().numpy().tolist()

    if args.task == 'binding_site':
        start = 0
        for data_id in data_dict.keys():
            prot_seq = data_dict[data_id]['prot_seq']
            length = len(prot_seq)
            probs = mean_preds[start:start + length]
            prot_binaries = binaries[start:start + length]
            data_dict[data_id].update({'probs': probs, 'binary': prot_binaries})
            start += length
    else:
        for i, data_id in enumerate(data_dict.keys()):
            print(data_id, preds[i] * args.scale + args.shift)
            data_dict[data_id].update({'delta_G': mean_preds[i] * args.scale + args.shift})
            data_dict[data_id].update({'Kd (nM)': delta_G_to_Kd(data_dict[data_id]['delta_G'])})
    
    if args.task == 'affinity':
        results = [['ID', 'delta G', 'Kd (nM)']]
        for key, value  in data_dict.items():
            results.append([key, value['delta_G'], value['Kd (nM)']])
            print([key, value['delta_G'], value['Kd (nM)']])
        check_dir(args.save_dir)

        with open(f'{args.save_dir}/{args.task}_results.csv', mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(results)
        torch.save(attns, f'{args.save_dir}/{args.task}_attns.pt')

    else:
        with open(f'{args.save_dir}/{args.task}_results.pkl', mode='wb') as file:
            pickle.dump(data_dict, file)