from argparse import ArgumentParser


BindingModelConfigs = {
    'node_input_dim': 580,
    'edge_input_dim': 450,
    'hidden_dim': 128,
    'num_layers': 4,
    'dropout': 0.2,
    'out_dim': 2,
    'pool': None
}


AffinityModelConfigs = {
    'node_input_dim': 570,
    'edge_input_dim': 450,
    'hidden_dim': 64,
    'num_layers': 1,
    'dropout': 0.4,
    'out_dim': 1,
    'pool': 'attn'
}


def parse_args():
    parser = ArgumentParser()

    #! Basic settings
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--cpu_num', type=int, default=8)
    parser.add_argument('--task', type=str, choices=['binding_site', 'affinity'], default='affinity')
    parser.add_argument('--input', type=str, default='./example_inputs')
    parser.add_argument('--save_dir', type=str, default='./results')
    
    #! Data
    parser.add_argument('--af3_exec_dir', type=str, default='/opt/data/private/alphafold/alphafold3')
    parser.add_argument('--af3_dir', type=str, default='./AF3_pred')
    parser.add_argument('--af3_max', type=float, default=500)
    parser.add_argument('--af3_min', type=float, default=-500)
    parser.add_argument('--radius', type=float, default=15.0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--shift', type=float, default=-15)
    parser.add_argument('--scale', type=float, default=10)
    
    #! Seed
    parser.add_argument('--seed', type=int, default=2025)
    
    args = parser.parse_args()
    if args.task == 'binding_site':
        for key, value in BindingModelConfigs.items():
            parser.add_argument(f'--{key}', default=value)
        compl = False
    else:
        for key, value in AffinityModelConfigs.items():
            parser.add_argument(f'--{key}', default=value)
        compl = True
        
    args = parser.parse_args()
    args.compl = compl
    return args