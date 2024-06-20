Implementation of the paper - Predicting Dose-Response Curves with Deep Neural Networks, published at [ICML 2024](https://icml.cc/virtual/2024/poster/34250). 
- `run_all_experiments.py` contains driver code for executing the experiments. 
- `train_model.py` contains the training loop for the model
- `optimize_hyperparameters.py` contains the hyperparameter optimization pipeline.
- in `utils.py` utilities for downloading and pre-processing the data are contained
- In FunFor the code from FunFor [1] found at https://github.com/xiaotiand/FunFor was adapted to support multi-threading and covariates with 0 variance.

1. Fu, G., Dai, X., & Liang, Y. (2021). Functional random forests for curve response. Scientific Reports, 11(1), 1-14.
