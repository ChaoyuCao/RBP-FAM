import os
import numpy as np
import torch
import torch.nn.functional as F
from Bio.PDB import MMCIFParser, PDBParser
from data.chemistry import AtomRadii, AtomMass
from data.chemistry import ResidueDictionary, RNALetters, DNALetters
from data.chemistry import RosettaParams
from data.chemistry import def_atom_features, rosetta_atom_type_map


atom_fea_dict = def_atom_features()


def check_protein_residue(residue):
    resname = residue.get_resname()
    if resname in ResidueDictionary.keys():
        return True
    else:
        return False


def check_protein_atom(atom):
    resname = atom.get_parent().get_resname()
    rosetta_valid = atom.get_name() in rosetta_atom_type_map[ResidueDictionary[resname]].keys()
    atom_valid = atom.get_name()[0] in AtomRadii.keys()
    return atom_valid and rosetta_valid


def residue_features(residue):
    resname = residue.get_resname()
    atom_features = []
    rosetta_params = []
    coords_dict = {'N': None, 'CA': None, 'C': None, 'O': None, 'R': []}
    for i, atom in enumerate(residue.get_atoms()):
        if not check_protein_atom(atom):
            continue
        atom_name = atom.get_name()
        atom_type = atom_name[0]
                
        #* chemical feature
        mass = AtomMass[atom_type]
        radius = AtomRadii[atom_type]
        bfactor = atom.get_bfactor()
        is_sidechain = 0 if atom_name in ['N','CA','C','O'] else 1
        try:
            #! charge, num_H, ring
            atom_chem_fea = atom_fea_dict[ResidueDictionary[resname]][atom_name]
        except KeyError:
            atom_chem_fea = [0.5, 0.5, 0.5]
        atom_feature = [mass / 32.0, 
                        radius / 1.8,
                        bfactor, is_sidechain] + atom_chem_fea
        atom_features.append(atom_feature)
        
        #* rosetta
        rosetta_type = rosetta_atom_type_map[ResidueDictionary[resname]][atom_name]
        rosetta_param = list(RosettaParams[rosetta_type].values())
        rosetta_params.append(rosetta_param)  #* list
        
        #* coords
        if not is_sidechain:
            coords_dict[atom_name] = atom.get_coord()
        else:
            coords_dict['R'].append(atom.get_coord())

    #* check missing value for N, CA, C and O
    for key, value in coords_dict.items():
        if key == 'R':
            continue
        if value is None:
            coords_dict[key] = [0.0, 0.0, 0.0]
    if len(coords_dict['R']) == 0:
        coords_dict['R'] = [coords_dict['CA']]

    #* get mean feature values as residue's atom features
    atom_features = np.array(atom_features)
    bfactors = atom_features[:, 2]
    max_bfactor, min_bfactor = np.max(bfactors), np.min(bfactors)
    if (max_bfactor - min_bfactor) == 0:
        res_bfactor = np.zeros(bfactors.shape) + 0.5
    else:
        res_bfactor = (bfactors - min_bfactor) / (max_bfactor - min_bfactor)
    atom_features = np.mean(np.array(atom_features), axis=0)
    atom_features[2] = np.mean(res_bfactor)
    
    #* mean rosetta_params
    rosetta_params = np.mean(rosetta_params, axis=0)

    #* merge coords to ndarray
    coords = [coords_dict['N'], coords_dict['CA'], coords_dict['C'], coords_dict['O'], np.mean(np.array(coords_dict['R']), axis=0)]
    coords = np.array(coords)  # (5, 3)
    return atom_features, rosetta_params, coords


def residue_coords(residue, mol_type):
    resname = residue.get_resname()
    if mol_type == 'prot':
        coords_dict = {'N': None, 'CA': None, 'C': None, 'O': None, 'R': []}
        for i, atom in enumerate(residue.get_atoms()):
            if not resname == 'UNK':
                if not check_protein_atom(atom):
                    continue
            atom_name = atom.get_name()
            atom_type = atom_name[0]
            is_sidechain = 0 if atom_name in ['N','CA','C','O'] else 1
            #* coords
            if not is_sidechain:
                coords_dict[atom_name] = atom.get_coord()
            else:
                coords_dict['R'].append(atom.get_coord())
        for key, value in coords_dict.items():
            if key == 'R':
                continue
            if value is None:
                coords_dict[key] = [0.0, 0.0, 0.0]
        if len(coords_dict['R']) == 0:
            coords_dict['R'] = [coords_dict['CA']]
        coords = [coords_dict['N'], coords_dict['CA'], coords_dict['C'], coords_dict['O'], np.mean(np.array(coords_dict['R']), axis=0)]
        
    else:
        # [C4', P, N1/N9, O5', C5']
        coords_dict = { "C4'": None, "P": None, 'N': None, "O5'": None, "C5'": None}
        for i, atom in enumerate(residue.get_atoms()):
            atom_name = atom.get_name().strip()
            if atom_name not in ["P", "C4'", "N1", "N9", "O5'", "C5'"]:
                continue
            if atom_name == 'N1' or atom_name == 'N9':
                atom_name = "N"
            coords_dict[atom_name] = atom.get_coord()

        for key, value in coords_dict.items():
            if value is None:
                coords_dict[key] = [0.0, 0.0, 0.0]
        coords = [coords_dict["C4'"], coords_dict["P"], coords_dict["N"], coords_dict["O5'"], coords_dict["C5'"]]
    return np.array(coords)


def read_single_protein(file_path, chain_id='A'):
    if file_path.endswith('.pdb'):
        parser = PDBParser(QUIET=True)
    else:
        parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure('struct', file_path)
    model = structure[0]
    chain = model[chain_id]
    residues = list(chain.get_residues())
    coords = []
    atom_features = []
    rosetta_params = []
    for residue in residues:     
        if not check_protein_residue(residue):
            continue
        atom_avg_fea, avg_rosetta_params, res_coords = residue_features(residue)
        atom_features.append(atom_avg_fea)
        rosetta_params.append(avg_rosetta_params)
        coords.append(res_coords)
    atom_features = np.array(atom_features)
    rosetta_params = np.array(rosetta_params)
    coords = np.array(coords)
    return atom_features, rosetta_params, coords


def read_compl(file_path):
    if file_path.endswith('.cif'):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    structure = parser.get_structure('struct', file_path)
    model = structure[0]
    coords = []
    chain = model['A']
    for residue in list(chain.get_residues()):
        res_coords = residue_coords(residue, 'prot')
        coords.append(res_coords)
    chain = model['B']            
    for residue in list(chain.get_residues()):
        res_coords = residue_coords(residue, 'rna')
        coords.append(res_coords)

    return np.array(coords)

