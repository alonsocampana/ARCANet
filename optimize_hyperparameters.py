from utils import download_file, CurveDataset, download_or_read, GDSC, CTRPv2, NCI60, PRISM, SmoothingSplitter, PrecisionOncologySplitter, ExtrapolationSplitter, InterpolationSplitter, DrugDiscoverySplitter, process_dataset
import os
import zipfile
import polars as pl
import pandas as pd
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, DataLoader
import torch
from torch import nn
import torchmetrics
from models_new import DeepSplineFusionModule, init_weights, CrossAttnPooling, LatentHillFusionModule, ConcatFusionModule, MLPEncoder, MultiModel, GatedAttnPooling
from torch.nn import functional as F
from torch_geometric import nn as gnn
import torch_geometric
from utils import build_model, metric_dict
from train_model import train_model
import optuna
import argparse

def train_model_optuna(trial, config):
    def optuna_step_callback(epoch, train_dict, test_dict):
        trial.report(test_dict["PearsonCorrCoef_test"], step = epoch)
        if np.isnan(train_dict["MeanSquaredError_train"]):
            raise optuna.TrialPruned()
        if train_dict["MeanSquaredError_train"] > 1:
            raise optuna.TrialPruned()
        if trial.should_prune():
            raise optuna.TrialPruned()
    max_iter = config["optimizer"]["max_iter"] + 0
    config["network"] = {"hidden_dim":trial.suggest_int("hidden_dim", 1, 4),
                "n_knots": 16,
                "n_pooling_heads": trial.suggest_int("n_pooling_heads", 1, 4),
                "dropout_cattn" : trial.suggest_float("dropout_cattn", 0.0, 0.50),
                "dropout_genes" : trial.suggest_float("dropout_genes", 0.0, 0.50),
                "dropout_fusion" : trial.suggest_float("dropout_fusion", 0.0, 0.50),
                "dropout_fc" :trial.suggest_float("dropout_fc", 0.0, 0.50),
                "dropout_nodes_attn" :trial.suggest_float("dropout_nodes_attn", 0.0, 0.50),
                "n_layers" :trial.suggest_int("n_layers", 1, 10),
                "n_transformers":trial.suggest_int("n_transformers", 0, 3),
                "n_heads" : trial.suggest_int("n_heads", 1, 4),
                "fc_hidden" :trial.suggest_int("fc_hidden", 256, 6000),
                "use_normalization" : trial.suggest_categorical("use_normalization", [True, False]),
                "use_normalization_fc" : trial.suggest_categorical("use_normalization_fc", [True, False]),
                "use_normalization_fusion": trial.suggest_categorical("use_normalization_fusion", [True, False]),
                "activation_fn": trial.suggest_categorical("activation_fn", ["relu", "sigmoid", "tanh"]),
                "fusion" :"hill",
                "transform_log_conc": trial.suggest_categorical("transform_log_conc", [True, False]),
                "crossattn" : "transformer"}
    config["optimizer"] = {"batch_size":256,
                      "learning_rate":trial.suggest_float("learning_rate", 0.00000001, 0.1, log=True),
                      "gamma_factor":0.5,
                      "alpha":trial.suggest_float("alpha", 0.00, 1),
                      "max_iter":max_iter,
                      "clip_norm":trial.suggest_float("clip_norm", 0.5, 10),}
    config["network"]["hidden_dim"]*=156
    try:
        return train_model(config, [optuna_step_callback])
    except Exception as e:
        print(e)
        return 0
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameters study")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="The dataset you want to use (a string)."
    )
    parser.add_argument(
        "--fold",
        type=int,
        required=True,
        help="The fold number (an integer)."
    )
    parser.add_argument(
        "--setting",
        type=str,
        required=True,
        help="The partition strategy you want to use."
    )
    parser.add_argument(
        "--cuda",
        type=int,
        required=True,
        help="The CUDA device to use (an integer)."
    )
    
    parser.add_argument(
        "--max_iter",
        type=int,
        required=True,
        help="Maximum number of epochs"
    )
    parser.add_argument(
        "--fingerprint",
        action = "store_true",
        help="Use fingerprint instead of GNN",
    )
    
    parser.add_argument(
        "--leave_out",
        type=int,
        required=False,
        default=None,
        help="Leaves one fold out for future testing"
    )
    
    
    args= parser.parse_args()
    fingerprint = args.fingerprint
    dataset = args.dataset
    fold = args.fold
    setting = args.setting
    cuda_device = args.cuda
    max_iter = args.max_iter
    config = {"env":{"dataset":dataset,
                "add_random_suffix":False,
                "setting":setting,
                "fold":fold,
                "leave_out":args.leave_out,
                "cuda_device":cuda_device,
                "debug":False,
                "mixed_precision":True,
                "missing_random":0.0,
                "missing_systematically":0.0,
                "fingerprint":fingerprint,
                "interpolation_augment":0.0},
             "optimizer": {"max_iter":max_iter}}
    objective = lambda x: train_model_optuna(x, config)
    if fingerprint:
        study_name = f"{dataset}_{setting}_fingerprint_{fold}"
    else:
        study_name = f"{dataset}_{setting}_{fold}"
    storage_name = "sqlite:///studies/{}.db".format(study_name)
    study = optuna.create_study(study_name=study_name,
                                storage=storage_name,
                                direction='maximize',
                                load_if_exists=True,
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=20,
                                                               n_warmup_steps=10,
                                                               interval_steps=10))
    study.optimize(objective, n_trials=100)