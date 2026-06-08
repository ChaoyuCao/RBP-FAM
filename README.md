# RBP-FAM
The code for RBP-FAM is provided for review and revision. It leverages AlphaFold 3-derived representations and graph neural networks (GNNs) to predict the binding sites and affinities of RNA-binding proteins. To reduce computational costs and enhance efficiency in this Capsule, we have included pre-computed AlphaFold 3-derived representations for execution. AlphaFold 3 -drived representations for all proteins used in this paper will be uploaded to Zenodo prior to publication.

## Installation
We recommend building two independent conda environments, named "af3" and "RBP-FAM". The "af3" environment is for running AlphaFold 3, and the "RBP-FAM" environment is for performing inference.

### AlphaFold 3
Please following the instruction of:
https://github.com/google-deepmind/alphafold3

### RBP-FAM
```
pip install numpy==2.4.0 torch==2.5.1 pandas==2.3.3 scipy==1.16.3 tqdm==4.67.1 biopython==1.86
pip install torch-scatter==2.1.2+pt25cu124 torch-sparse==0.6.18+pt25cu124 torch-cluster==1.6.3+pt25cu124 torch-spline-conv==1.2.2+pt25cu124 -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
pip install torch-geometric==2.6.1
```

## Inference
To run inference in this Capsule, juse use run.sh.

To inference with calculating AlphaFold 3 representations:
```
#! /bin/bash
# 
eval "$(conda shell.bash hook)"

conda activate af3

python alphafold3.py --task binding_site --input example_inputs/9natcom.fasta --af3_exec_dir your_af3_exec_dir
python alphafold3.py --task affinity --input example_inputs/9natcom.csv --af3_exec_dir your_af3_exec_dir

conda deactivate

conda activate RBP-FAM

python inference.py --task binding_site --input example_inputs/9natcom.fasta --save_dir results/binding
python inference.py --task affinity --input example_inputs/9natcom.fasta --save_dir results/affinity
```
