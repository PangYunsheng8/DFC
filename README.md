# DFC
DFC code for paper 
>DeepFuzzyClusteringwithWeightedIntra-classVarianceandExtendedMutual InformationRegularization
## Requirement
* python 3.5
* munkres
* argparse
* pickle
* sklearn
* numpy==1.14.1
* tensorflow==1.9.0
## Files
* DFC.py is the implementation of DFC algorithm
* File data contains data makers
* File centers contains initialization centers
* File pretrain contains pretrain checkpoint obtained by an auto-encoder
## Run DFC on Reuters
* Run DFC on Reuters Dataset
```
python DFC.py
```

