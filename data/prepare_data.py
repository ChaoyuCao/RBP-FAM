import numpy as np
import os
from Bio import PDB
from Bio.PDB.PDBParser import PDBParser
import torch
from data.featurizer import *
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import coalesce
from torch_geometric.loader import DataLoader
from torch_geometric.nn import radius_graph, knn_graph
import pickle
import sys


def lst_to_tensor(lst, dtype_=torch.float32):
    tensors = []
    for array in lst:
        if not isinstance(array, torch.Tensor):
            tensor = torch.tensor(array, dtype=dtype_)
        else:
            tensor = array
        tensors.append(tensor)
    return tensors


class DataGenerator(Dataset):
    def __init__(self, args, data_dict):
        self.args = args
        self.radius = args.radius
        self.compl = args.compl
        self.af3_dir = args.af3_dir
        self.data_dict = data_dict
        self.af3_min = args.af3_min
        self.af3_max = args.af3_max
        self.data_ids = list(self.data_dict.keys())
        self.device = args.device
        
    def __len__(self):
        return len(list(self.data_dict.keys()))
    
    def __getitem__(self, idx):
        if not self.compl:
            return self.build_graph(idx)
        else:
            return self.build_compl_graph(idx)
        
    def get_af3(self, data_id, length):
        af3_path = f'{self.af3_dir}/structs/{data_id.lower()}/predicted/{data_id.lower()}/seed-1_embeddings/embeddings.npz'
        af3_feat = np.load(af3_path)
        prot_repr = af3_feat['single_embeddings']
        repr_size = prot_repr.shape[0]
        if repr_size > length:
            prot_repr = prot_repr[:length, :]
        prot_repr = np.clip(prot_repr, self.af3_min, self.af3_max)
        prot_repr = (prot_repr - self.af3_min) / (self.af3_max - self.af3_min)
        return prot_repr
    
    def load_features(self, data_id, seq):
        prot_length = len(seq)
        prot_repr = self.get_af3(data_id, prot_length)
        cif_path = f'{self.af3_dir}/structs/{data_id.lower()}/predicted/{data_id.lower()}/{data_id.lower()}_model.cif'

        atom_features, rosetta_params, coords = read_single_protein(cif_path)
   
        prot_repr, atom_features, rosetta_params, coords = lst_to_tensor(
            [prot_repr, atom_features, rosetta_params, coords]
        )
        
        return prot_repr, atom_features, rosetta_params, coords
    
    def load_compl_features(self, data_id):
        data = self.data_dict[data_id]
        prot_len = len(data['prot_seq'])
        rna_len = len(data['rna_seq'])
        total_len = prot_len + rna_len
        
        single_repr = self.get_af3(data_id, total_len)
        
        cif_path = f'{self.af3_dir}/structs/{data_id.lower()}/predicted/{data_id.lower()}/{data_id.lower()}_model.cif'
        coords = read_compl(cif_path)
        
        mol_mask = torch.zeros(total_len, )
        mol_mask[prot_len:] = 1.0 
        
        bindings = search_binding(cif_path)
        binding_mask = []
        for chain_binding in bindings:
            binding_mask += chain_binding
        binding_mask = np.array(binding_mask)
        
        single_repr, coords, binding_mask = lst_to_tensor([single_repr, coords, binding_mask])
        return single_repr, coords, binding_mask, mol_mask
    
    @torch.no_grad()
    def build_graph(self, idx):
        pdb_id = self.data_ids[idx]
        seq = self.data_dict[pdb_id]['prot_seq']
        (
            prot_repr,
            atom_features, rosetta_params,
            coords
        ) = self.load_features(pdb_id, seq)
        
        all_node_features = {'prot_repr': prot_repr,
                             'atom_features': atom_features, 
                             'rosetta_params': rosetta_params}
        
        node_features = torch.cat(list(all_node_features.values()), dim=-1)

        calpha = coords[:, 1]

        edge_index = radius_graph(calpha, r=self.radius, loop=True, 
                                  max_num_neighbors=1000, num_workers=8)

        graph_data = Data(name=pdb_id, X=coords, node_feat=node_features, edge_index=edge_index, mask=torch.ones_like(node_features[:, 0]))
        return graph_data.to(self.device)
    
    @torch.no_grad()
    def build_compl_graph(self, idx):
        data_id = self.data_ids[idx]
        single_repr, coords, binding_mask, mol_mask = self.load_compl_features(data_id)
        node_features = torch.cat([single_repr, mol_mask[:, None], binding_mask[:, None]], dim=-1)
        
        c_core = coords[:, 1]  # C_alpha and P
        c_alpha = c_core[mol_mask.bool()]
        p = c_core[~mol_mask.bool()]
        dist_map = torch.sqrt(torch.sum((c_alpha[:, None] - p[None])**2, dim=-1))
        prot_dist = torch.min(dist_map, dim=-1).values
        rna_dist = torch.min(dist_map, dim=0).values
        dist = torch.cat([prot_dist, rna_dist], dim=0)
        dist = 1 - dist / torch.max(dist) + 1e-6
        edge_index = radius_graph(c_core, r=self.radius, loop=True, 
                                  max_num_neighbors=1000, num_workers=8)
    
        graph_data = Data(name=data_id, X=coords, node_feat=node_features, edge_index=edge_index, mask=dist)
        return graph_data.to(self.device)
    
    
def get_atoms(residue):
    #! residue can be residue or chain
    coords = []
    radii = []
    for atom in residue.get_atoms():
        atom_type = atom.get_name()[0]
        if atom_type not in AtomRadii.keys():
            continue
        vdw_radius = AtomRadii[atom_type]
        coord = atom.get_coord()
        coords.append(coord)
        radii.append(vdw_radius)
    coords = np.array(coords)
    radii = np.array(radii)
    return coords, radii


def check_binding(prot_chain, rna_chain):
    #! get atoms from rna_chain
    rna_coords, rna_radii = get_atoms(rna_chain)
    #! check residue binding
    binding_res = []
    for residue in prot_chain.get_residues():
        res_coords, res_radii = get_atoms(residue)
        #* res_coords: (N, 3), rna_coords: (M, 3)
        diff = res_coords[:, None] - rna_coords[None]
        dists = np.sqrt(np.sum(diff**2, axis=-1))  # (N, M)
        min_dists = np.min(dists, axis=-1) # (N)
        min_rna_atom_indices = np.argmin(dists, axis=-1)
        rna_radii_min = rna_radii[min_rna_atom_indices]  # (N)
        vdw_dists = min_dists - res_radii - rna_radii_min
        atom_bindings = vdw_dists < 0.5
        if atom_bindings.any():
            binding_res.append(f'{residue.get_resname()}_{residue.get_id()[1]}')
    return binding_res

    
def search_binding(file_path, prot_id='A', rna_id='B'):
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(file_path, file_path)
    model = structure[0]
    chain_ids = [prot_id, rna_id]
    other_chain_ids = [prot_id, rna_id]
    chain_bindings = []
    for target_chain_id in chain_ids:
        target_chain = model[target_chain_id]
        all_binding_residues = []
        for other_chain_id in other_chain_ids:
            if other_chain_id == target_chain_id:
                continue
            other_chain = model[other_chain_id]
            binding_res = check_binding(target_chain, other_chain)
            if len(binding_res) > 0:
                all_binding_residues += binding_res
        all_binding_residues = list(set(all_binding_residues))
        
        binding = []
        for residue in target_chain.get_residues():
            resname = residue.get_resname()
            full_resname = f'{resname}_{residue.get_id()[1]}'
            if full_resname in all_binding_residues:
                binding.append(1)
            else:
                binding.append(0)
        chain_bindings.append(binding)
    return chain_bindings