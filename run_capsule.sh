#! /bin/bash
# 
eval "$(conda shell.bash hook)"

python inference.py --task binding_site --input example_inputs/9natcom.fasta --save_dir results/binding
python inference.py --task affinity --input example_inputs/9natcom.csv --save_dir results/affinity