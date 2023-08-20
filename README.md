# TypicalityCoding

Simple example of how to use an embedding plus sphering transform to measure document typicality with respect to a corpus.

To install environment:
    conda env create --file=tc_env_def.yaml

Specific versions used saved as:
    conda list --export > tc_env_list.txt

To use environment:
    conda activate tc_env

Download data from:
    https://drive.google.com/drive/folders/1sXtDqNF-D0mypORzfevir_eH9FdG2668?usp=sharing
    and then unzip 

To run tests:
    python -m pytest tests

