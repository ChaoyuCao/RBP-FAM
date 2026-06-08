import numpy as np


"""
Letters
"""
DNALetters = ['DA', 'DC', 'DG', 'DT']
RNALetters = ['A', 'C', 'G', 'U', 'OMA', 'OMC', 'OMG', 'OMU']


"""
Radii
"""
#! From MaSIF: see https://github.com/LPDI-EPFL/masif/blob/master/source/default_config/chemistry.py
#! Van der Waals radii
AtomRadii = {'N': 1.55,
             'O': 1.52,
             'C': 1.70,
             'H': 1.20,
             'S': 1.80,
             'P': 1.80}


"""
Mass
"""
AtomMass = {'N': 14,
            'O': 16,
            'C': 12,
            'S': 32,
            'H': 1,
            'P': 31}


"""
Residue and atom types
"""
ResidueDictionary = {'CYS': 'C', 
                     'ASP': 'D', 
                     'SER': 'S', 'SEP': 'S',
                     'GLN': 'Q', 
                     'LYS': 'K',
                     'ILE': 'I', 
                     'PRO': 'P', 
                     'THR': 'T', 'TPO': 'T',
                     'PHE': 'F', 
                     'ASN': 'N',
                     'GLY': 'G', 
                     'HIS': 'H',
                     'LEU': 'L', 
                     'ARG': 'R', 
                     'TRP': 'W',
                     'ALA': 'A', 
                     'VAL': 'V', 
                     'GLU': 'E', 
                     'TYR': 'Y', 'PTR': 'Y',
                     'MET': 'M', 'MSE': 'M'}

hetresidue_field = [' '] + ['H_%s'%name for name in ResidueDictionary.keys()]

AtomDictionary = {'C': 'C', 'CA': 'C', 'CB': 'C',
                  'CD': 'C', 'CD1': 'C', 'CD2': 'C',
                  'CE': 'C', 'CE1': 'C', 'CE2': 'C',
                  'CE3': 'C', 'CG': 'C', 'CG1': 'C',
                  'CG2': 'C', 'CH2': 'C', 'CZ': 'C',
                  'CZ2': 'C', 'CZ3': 'C', 'N': 'N',
                  'ND1': 'N', 'ND2': 'N', 'NE': 'N',
                  'NE1': 'N', 'NE2': 'N', 'NH1': 'N',
                  'NH2': 'N', 'NZ': 'N', 'O': 'O',
                  'OD1': 'O', 'OD2': 'O', 'OE1': 'O',
                  'OE2': 'O', 'OG': 'O', 'OG1': 'O',
                  'OH': 'O', 'OXT': 'O', 'SD': 'S',
                  'SG': 'S', 'H': 'H'}

def map_rosetta_to_pdb_atom_types():
     A = {'N': 'Nbb', 'CA': 'CAbb', 'C': 'CObb', 'O': 'OOC', 'CB': 'CH3', 'OXT': 'OOC',
          'H': 'HNbb', 'HA': 'Hapo', 'HB1': 'Hapo', 'HB2': 'Hapo', 'HB3': 'Hapo'}
     
     V = {'N': 'Nbb', 'CA': 'CAbb', 'C': 'CObb', 'O': 'OCbb',  'OXT': 'OOC',
          'CB': 'CH1', 'CG1': 'CH3', 'CG2': 'CH3',
          'H': 'HNbb', 'HA': 'Hapo', 'HB': 'Hapo', 'HG11': 'Hapo', 'HG12': 'Hapo', 'HG13': 'Hapo', 'HG21': 'Hapo', 
          'HG22': 'Hapo', 'HG23': 'Hapo'}
     
     F = {'N': 'Nbb', 'CA': 'CAbb', 'C': 'CObb', 'O': 'OCbb', 'CB': 'CH2',  'OXT': 'OOC',
          'CG': 'aroC', 'CD1': 'aroC', 'CD2': 'aroC', 'CE1': 'aroC', 'CE2': 'aroC', 'CZ': 'aroC',
          'H': 'HNbb', 'HA': 'Hapo', 'HB2': 'Hapo', 'HB3': 'Hapo', 
          'HD1': 'Haro', 'HD2': 'Haro', 'HE1': 'Haro', 'HE2': 'Haro', 'HZ': 'Haro'}
     
     P = {'N': 'Npro', 'CA': 'CAbb', 'C': 'CObb', 'O': 'OCbb', 'OXT': 'OOC',
          'CB': 'CH2', 'CG': 'CH2', 'CD': 'CH2', 'H': 'Hapo', #! No other H types besides Hapo was found in Pro
          'HA': 'Hapo', 'HB2': 'Hapo', 'HB3': 'Hapo', 'HG2': 'Hapo', 'HG3': 'Hapo', 'HD2': 'Hapo', 'HD3': 'Hapo'}
     
     L = {'N': 'Nbb', 'CA': 'CAbb', 'C': 'CObb', 'O': 'OCbb',  'OXT': 'OOC',
          'CB': 'CH2', 'CG': 'CH1', 'CD1': 'CH3', 'CD2': 'CH3',
          'H': 'HNbb', 'HA': 'Hapo', 'HB2': 'Hapo', 'HB3': 'Hapo', 'HG': 'Hapo', 
          'HD11': 'Hapo', 'HD12': 'Hapo', 'HD13': 'Hapo', 'HD21': 'Hapo', 'HD22': 'Hapo', 'HD23': 'Hapo'}
     
     I = {'N': 'Nbb', 'CA': 'CAbb', 'C': 'CObb', 'O': 'OCbb', 'CB': 'CH1',
          'CG1': 'CH2', 'CG2': 'CH3', 'CD1': 'CH3', 'OXT': 'OOC',
          'H':'HNbb', 'HA': 'Hapo', 'HB': 'Hapo', 'HG12': 'Hapo', 'HG13': 'Hapo', 'HG21': 'Hapo', 'HG22': 'Hapo', 
          'HG23': 'Hapo', 'HD11': 'Hapo', 'HD12': 'Hapo', 'HD13': 'Hapo'}
     
     R = {'N': 'Nbb', 'CA': 'CAbb', 'C': 'CObb', 'O': 'OCbb', 'CB': 'CH2', 'OXT': 'OOC',
          'CG': 'CH2', 'CD': 'CH2', 'NE': 'Narg', 'CZ': 'aroC', 'NH1': 'Narg', 'NH2': 'Narg',
          'H': 'HNbb', 'HE': 'Hapo', 'HH11': 'Hpol', 'HH12': 'Hpol', 'HH21': 'Hpol', 'HH22': 'Hpol', 
          'HA': 'Hapo', 'HB2': 'Hapo', 'HB3': 'Hapo', 'HG2': 'Hapo', 'HG3': 'Hapo', 'HD2': 'Hapo', 'HD3': 'Hapo'}
     
     D = {'N': 'Nbb', 'CA': 'CAbb', 'C': 'CObb', 'O': 'OCbb', 'OXT': 'OOC',
          'CB': 'CH2', 'CG': 'COO', 'OD1': 'OOC', 'OD2': 'OOC',
          'H': 'HNbb', 'HA': 'Hapo', 'HB2': 'Hapo', 'HB3': 'Hapo', 'HD2': 'Hapo'}
          
     E = {'N': 'Nbb', 'CA': 'CAbb', 'C': 'CObb', 'O': 'OCbb', 'OXT': 'OOC',
          'CB': 'CH2', 'CG': 'CH2', 'CD': 'COO', 'OE1': 'OOC', 'OE2': 'OOC',
          'H': 'HNbb', 'HA': 'Hapo', 'HB2': 'Hapo', 'HB3': 'Hapo', 'HG2': 'Hapo', 'HG3': 'Hapo', 'HE2': 'Hapo'}

     S = {'N': 'Nbb', 'CA': 'CAbb', 'C': 'CObb', 'O': 'OCbb', 'CB': 'CH2', 'OG': 'OH', 'OXT': 'OOC',
          'H': 'HNbb', 'HG': 'Hpol', 'HA': 'Hapo', 'HB2': 'Hapo', 'HB3': 'Hapo', 'H1': 'HNbb', 'H2': 'HNbb', 'H3': 'HNbb'}
     
     T = {'N': 'Nbb', 'CA': 'CAbb', 'C': 'CObb', 'O': 'OCbb',
          'CB': 'CH1', 'OG1': 'OH', 'CG2': 'CH3', 'OXT': 'OOC',
          'H': 'HNbb', 'HG1': 'Hpol', 'HA': 'Hapo', 'HB': 'Hapo', 'HG21': 'Hapo', 'HG22': 'Hapo', 'HG23': 'Hapo'}
     
     C = {'N': 'Nbb', 'CA': 'CAbb', 'C': 'CObb', 'O': 'OCbb', 'CB': 'CH2', 'SG': 'S', 'OXT': 'OOC',
          'H': 'HNbb', 'HA': 'Hapo', 'HB2': 'Hapo', 'HB3': 'Hapo', 'HG': 'Hapo'}
     
     N = {'N': 'Nbb', 'CA': 'CAbb', 'C': 'CObb', 'O': 'OCbb', 'OXT': 'OOC',
          'CB': 'CH2', 'CG': 'CNH2', 'OD1': 'ONH2', 'ND2': 'NH2O',
          'H': 'HNbb', 'HD21': 'Hpol', 'HD22': 'Hpol', 'HA': 'Hapo', 'HB2': 'Hapo', 'HB3': 'Hapo'}
     
     Q = {'N': 'Nbb', 'CA': 'CAbb', 'C': 'CObb', 'O': 'OCbb', 'CB': 'CH2', 'OXT': 'OOC',
          'CG': 'CH2', 'CD': 'CNH2', 'OE1': 'ONH2', 'NE2': 'NH2O',
          'H': 'HNbb', 'HE21': 'Hpol', 'HE22': 'Hpol', 'HA': 'Hapo', 'HB2': 'Hapo', 'HB3': 'Hapo', 'HG2': 'Hapo', 'HG3': 'Hapo'}
     
     H = {'N': 'Nbb', 'CA': 'CAbb', 'C': 'CObb', 'O': 'OCbb', 'CB': 'CH2', 'OXT': 'OOC',
          'CG': 'aroC', 'ND1': 'Nhis', 'CD2': 'aroC', 'CE1': 'aroC', 'NE2': 'Ntrp',
          'H': 'HNbb', 'HD1': 'Hapo', 'HD2': 'Hapo', 'HA': 'Hapo', 'HB2': 'Hapo', 'HB3': 'Hapo', 'HE1': 'Hpol', 'HE2': 'Hpol'}
     
     K = {'N': 'Nbb', 'CA': 'CAbb', 'C': 'CObb', 'O': 'OCbb', 'CB': 'CH2', 'CG': 'CH2', 'CD': 'CH2', 
          'CE': 'CH2', 'NZ': 'Nlys', 'OXT': 'OOC',
          'H': 'HNbb', 'HA': 'Hapo', 'HB2': 'Hapo', 'HB3': 'Hapo', 'HG2': 'Hapo', 'HG3': 'Hapo', 
          'HD2': 'Hapo', 'HD3': 'Hapo', 'HE2': 'Hapo', 'HE3': 'Hapo', 'HZ1': 'Hpol', 'HZ2': 'Hpol', 'HZ3': 'Hpol'}
     
     Y = {'N': 'Nbb', 'CA': 'CAbb', 'C': 'CObb', 'O': 'OCbb', 'CB': 'CH2', 'OXT': 'OOC',
          'CG': 'aroC', 'CD1': 'aroC', 'CD2': 'aroC', 'CE1': 'aroC', 'CE2': 'aroC', 'CZ': 'aroC', 'OH': 'OH',
          'H': 'HNbb', 'HH': 'Hpol', 'HA': 'Hapo', 'HB2': 'Hapo', 'HB3': 'Hapo', 
          'HD1': 'Haro', 'HD2': 'Haro', 'HE1': 'Haro', 'HE2': 'Haro'}
     
     M = {'N': 'Nbb', 'CA': 'CAbb', 'C': 'CObb', 'O': 'OCbb', 'CB': 'CH2', 'CG': 'CH2', 'SD': 'S', 'CE': 'CH3', 'OXT': 'OOC',
          'H': 'HNbb', 'HA': 'Hapo', 'HB2': 'Hapo', 'HB3': 'Hapo', 'HG2': 'Hapo', 'HG3': 'Hapo', 
          'HE1': 'Hapo', 'HE2': 'Hapo', 'HE3': 'Hapo', 'H1': 'Hapo', 'H2': 'Hapo', 'H3': 'Hapo', 'P': 'Phos'}
     
     W = {'N': 'Nbb', 'CA': 'CAbb', 'C': 'CObb', 'O': 'OCbb', 'CB': 'CH2', 'OXT': 'OOC',
          'CG': 'aroC', 'CD1': 'aroC', 'CD2': 'aroC', 'NE1': 'Ntrp', 'CE2': 'aroC', 
          'CE3': 'aroC', 'CZ2': 'aroC', 'CZ3': 'aroC', 'CH2': 'aroC',
          'H': 'HNbb', 'HE1': 'Haro', 'HA': 'Hapo', 'HB2': 'Hapo', 'HB3': 'Hapo', 'HD1': 'Haro', 
          'HE3': 'Haro', 'HZ2': 'Haro', 'HZ3': 'Haro', 'HH2': 'Hpol'}
     
     G = {'N': 'Nbb', 'CA': 'CAbb', 'C': 'CObb', 'O': 'OCbb', 'OXT': 'OOC',
          'H': 'HNbb', 'HA2': 'Hapo', 'HA3': 'Hapo'}
     
     rosetta_atom_type_map = {'A': A, 'V': V, 'F': F, 'P': P, 'L': L, 'I': I, 'R': R, 'D': D, 'E': E, 'S': S,
                              'T': T, 'C': C, 'N': N, 'Q': Q, 'H': H, 'K': K, 'Y': Y, 'M': M, 'W': W, 'G': G}
     
     for key, value in rosetta_atom_type_map.items():
          for H_name in ['H1', 'H2', 'H3']:
               if not H_name in value.keys():
                    value[H_name] = 'Hapo'
          rosetta_atom_type_map[key] = value
     return rosetta_atom_type_map

rosetta_atom_type_map = map_rosetta_to_pdb_atom_types()

"""
Rosetta Energy function terms 
"""
#! From rosetta.source.release-362\main\database\chemical\atom_type_sets\fa_standard\atom_properties.txt
RosettaParams = {'CAbb': {'lj_radius': 2.011760, 'well_depth': 0.062642, 'deltaG': 2.533791, 'lk_lambda': 3.5000, 'lk_vol': 12.1370},
                 'CNH2': {'lj_radius': 1.968297, 'well_depth': 0.094638, 'deltaG': 3.077030, 'lk_lambda': 3.5000, 'lk_vol': 13.5000},
                 'CH0': {'lj_radius': 2.011760, 'well_depth': 0.062642, 'deltaG': 1.409284, 'lk_lambda': 3.5000, 'lk_vol': 8.9980},
                 'CH1': {'lj_radius': 2.011760, 'well_depth': 0.062642, 'deltaG': -3.538387, 'lk_lambda': 3.5000, 'lk_vol': 10.6860},
                 'CH2': {'lj_radius': 2.011760, 'well_depth': 0.062642, 'deltaG': -1.854658, 'lk_lambda': 3.5000, 'lk_vol': 18.3310},
                 'CH3': {'lj_radius': 2.011760, 'well_depth': 0.062642, 'deltaG': 7.292929, 'lk_lambda': 3.5000, 'lk_vol': 25.8550},
                 'COO': {'lj_radius': 1.916661, 'well_depth': 0.141799, 'deltaG': -3.332648, 'lk_lambda': 3.5000, 'lk_vol': 14.6530},
                 'CObb': {'lj_radius': 1.916661, 'well_depth': 0.141799, 'deltaG': 3.104248, 'lk_lambda': 3.5000, 'lk_vol': 13.2210},
                 'aroC': {'lj_radius': 2.016441, 'well_depth': 0.068775, 'deltaG': 1.797950, 'lk_lambda': 3.5000, 'lk_vol': 16.7040},
                 'NH2O': {'lj_radius': 1.802452, 'well_depth': 0.161725, 'deltaG': -8.101638, 'lk_lambda': 3.5000, 'lk_vol': 15.6890},
                 'Narg': {'lj_radius': 1.802452, 'well_depth': 0.161725, 'deltaG': -8.968351, 'lk_lambda': 3.5000, 'lk_vol': 15.7170},
                 'Nbb': {'lj_radius': 1.802452, 'well_depth': 0.161725, 'deltaG': -9.969494, 'lk_lambda': 3.5000, 'lk_vol': 15.9920},
                 'Nhis': {'lj_radius': 1.802452, 'well_depth': 0.161725, 'deltaG': -9.739606, 'lk_lambda': 3.5000, 'lk_vol': 9.3177},
                 'Nlys': {'lj_radius': 1.802452, 'well_depth': 0.161725, 'deltaG': -20.864641, 'lk_lambda': 3.5000, 'lk_vol': 16.5140},
                 'Npro': {'lj_radius': 1.802452, 'well_depth': 0.161725, 'deltaG': -0.984585, 'lk_lambda': 3.5000, 'lk_vol': 3.7181},                 
                 'Ntrp': {'lj_radius': 1.802452, 'well_depth': 0.161725, 'deltaG': -8.413116, 'lk_lambda': 3.5000, 'lk_vol': 9.5221},                 
                 'OCbb': {'lj_radius': 1.540580, 'well_depth': 0.142417, 'deltaG': -8.006829, 'lk_lambda': 3.5000, 'lk_vol': 12.1960},                 
                 'OOC': {'lj_radius': 1.492871, 'well_depth': 0.099873, 'deltaG': -9.239832, 'lk_lambda': 3.5000, 'lk_vol': 9.9956},                 
                 'OH': {'lj_radius': 1.542743, 'well_depth': 0.161947, 'deltaG': -8.133520, 'lk_lambda': 3.5000, 'lk_vol': 10.7220},              
                 'ONH2': {'lj_radius': 1.548662, 'well_depth': 0.182924, 'deltaG': -6.591644, 'lk_lambda': 3.5000, 'lk_vol': 10.1020},
                 'S': {'lj_radius': 1.975967, 'well_depth': 0.455970, 'deltaG': -1.707229, 'lk_lambda': 3.5000, 'lk_vol': 17.640000},
                 'SH1': {'lj_radius': 1.975967, 'well_depth': 0.455970, 'deltaG': 3.291643, 'lk_lambda': 3.5000, 'lk_vol': 23.240000},
                 'HNbb': {'lj_radius': 0.901681, 'well_depth': 0.005000, 'deltaG': 0.0000, 'lk_lambda': 3.5000, 'lk_vol': 0.0000},
                 'HS': {'lj_radius': 0.363887, 'well_depth': 0.050836, 'deltaG': 0.0000, 'lk_lambda': 3.5000, 'lk_vol': 0.0000},
                 'Hapo': {'lj_radius': 1.421272, 'well_depth': 0.021808, 'deltaG': 0.0000, 'lk_lambda': 3.5000, 'lk_vol': 0.0000},
                 'Haro': {'lj_radius': 1.374914, 'well_depth': 0.015909, 'deltaG': 0.0000, 'lk_lambda': 3.5000, 'lk_vol': 0.0000},
                 'Hpol': {'lj_radius': 0.901681, 'well_depth': 0.005000, 'deltaG': 0.0000, 'lk_lambda': 3.5000, 'lk_vol': 0.0000},
                 'Phos': {'lj_radius': 2.150000, 'well_depth': 0.585000, 'deltaG': -24.0000, 'lk_lambda': 3.5000, 'lk_vol': 34.8000}}


"""
Other features
"""
#! From GraphBind
def def_atom_features():
    A = {'N':[0,1,0], 'CA':[0,1,0], 'C':[0,0,0], 'O':[0,0,0], 'CB':[0,3,0]}
    V = {'N':[0,1,0], 'CA':[0,1,0], 'C':[0,0,0], 'O':[0,0,0], 'CB':[0,1,0], 'CG1':[0,3,0], 'CG2':[0,3,0]}
    F = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0],'CB':[0,2,0],
         'CG':[0,0,1], 'CD1':[0,1,1], 'CD2':[0,1,1], 'CE1':[0,1,1], 'CE2':[0,1,1], 'CZ':[0,1,1] }
    P = {'N': [0, 0, 1], 'CA': [0, 1, 1], 'C': [0, 0, 0], 'O': [0, 0, 0],'CB':[0,2,1], 'CG':[0,2,1], 'CD':[0,2,1]}
    L = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'CG':[0,1,0], 'CD1':[0,3,0], 'CD2':[0,3,0]}
    I = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,1,0], 'CG1':[0,2,0], 'CG2':[0,3,0], 'CD1':[0,3,0]}
    R = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0],
         'CG':[0,2,0], 'CD':[0,2,0], 'NE':[0,1,0], 'CZ':[1,0,0], 'NH1':[0,2,0], 'NH2':[0,2,0] }
    D = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'CG':[-1,0,0], 'OD1':[-1,0,0], 'OD2':[-1,0,0]}
    E = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'CG':[0,2,0], 'CD':[-1,0,0], 'OE1':[-1,0,0], 'OE2':[-1,0,0]}
    S = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'OG':[0,1,0]}
    T = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,1,0], 'OG1':[0,1,0], 'CG2':[0,3,0]}
    C = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'SG':[-1,1,0]}
    N = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'CG':[0,0,0], 'OD1':[0,0,0], 'ND2':[0,2,0]}
    Q = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'CG':[0,2,0], 'CD':[0,0,0], 'OE1':[0,0,0], 'NE2':[0,2,0]}
    H = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0],
         'CG':[0,0,1], 'ND1':[-1,1,1], 'CD2':[0,1,1], 'CE1':[0,1,1], 'NE2':[-1,1,1]}
    K = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'CG':[0,2,0], 'CD':[0,2,0], 'CE':[0,2,0], 'NZ':[0,3,1]}
    Y = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0],
         'CG':[0,0,1], 'CD1':[0,1,1], 'CD2':[0,1,1], 'CE1':[0,1,1], 'CE2':[0,1,1], 'CZ':[0,0,1], 'OH':[-1,1,0]}
    M = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'CG':[0,2,0], 'SD':[0,0,0], 'CE':[0,3,0]}
    W = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0],
         'CG':[0,0,1], 'CD1':[0,1,1], 'CD2':[0,0,1], 'NE1':[0,1,1], 'CE2':[0,0,1], 'CE3':[0,1,1], 'CZ2':[0,1,1], 'CZ3':[0,1,1], 'CH2':[0,1,1]}
    G = {'N': [0, 1, 0], 'CA': [0, 2, 0], 'C': [0, 0, 0], 'O': [0, 0, 0]}

    atom_features = {'A': A, 'V': V, 'F': F, 'P': P, 'L': L, 'I': I, 'R': R, 'D': D, 'E': E, 'S': S,
                     'T': T, 'C': C, 'N': N, 'Q': Q, 'H': H, 'K': K, 'Y': Y, 'M': M, 'W': W, 'G': G}
    
    for atom_fea in atom_features.values():
        for i in atom_fea.keys():
            i_fea = atom_fea[i]
            atom_fea[i] = [i_fea[0] / 2 + 0.5, i_fea[1] / 3, i_fea[2]]

    return atom_features
