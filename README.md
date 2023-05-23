# ARCANet

Code for the submission "Predicting Dose-Response Curves with Graph-Attentional Networks and Variational Curve Representations"

## Baselines

The `FunFor` folder contains the code from Fu, G., Dai, X., & Liang, Y. (2021). Functional random forests for curve response. Scientific Reports, 11(1), 1-14 adapted from https://github.com/xiaotiand/FunFor
The `R` folder contains the code from  Vis, D.J. et al. Pharmacogenomics 2016, 17(7):691-700 adapted from https://github.com/CancerRxGene/gdscIC50

## Models
`models.py` contains the implementations of ARCANet
`OptimizeHyperpar.py` contains the script used for hyperparameter optimization
`run_all_experiments.sh` contains the driver code for executing all the experiments