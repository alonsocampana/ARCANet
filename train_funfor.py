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
from models_new import DeepSplineFusionModule, init_weights, CrossAttnPooling, LatentHillFusionModule, ConcatFusionModule, MLPEncoder, MultiModel, GatedAttnPooling, GATRes
from torch.nn import functional as F
from torch_geometric import nn as gnn
import torch_geometric
from utils import build_model, build_model_fingerprint, metric_dict, print_epoch, get_train_test_data, serialize_config, get_tabular
import argparse
import uuid
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr



def train_funfor(config):
    (X_train, y_train), (X_test, y_test) = get_tabular(config["env"]["dataset"],
                config["env"]["setting"],
                config["env"]["fold"])
    code = str(uuid.uuid4())
    keep_var = X_train.var(axis=0) > 0.015
    X_train = X_train[:, keep_var]
    X_test = X_test[:, keep_var]
    pd.DataFrame(X_train).to_csv(f"temp/temp_{code}_Xtrain.csv", index=False)
    pd.DataFrame(X_test).to_csv(f"temp/temp_{code}_Xtest.csv", index=False)
    pd.DataFrame(y_train).to_csv(f"temp/temp_{code}_ytrain.csv", index=False)
    pd.DataFrame(y_test).to_csv(f"temp/temp_{code}_ytest.csv", index=False)
    command = f"Rscript FunFor/fit_data.R {code}"
    os.system(command)
    y_pred = pd.read_csv(f"temp/temp_{code}_ypred.csv", index_col=0).iloc[1:].to_numpy().squeeze()
    r = pearsonr(y_pred.flatten(), y_test.flatten())[0]
    m = mean_squared_error(y_pred.flatten(), y_test.flatten())
    pd.Series({"MeanSquaredError_test" :m, "PearsonCorrCoef_test" :r}).to_csv(f"results_baseline/FunFor_{config['env']['dataset']}_{config['env']['setting']}_{config['env']['fold']}.csv")
    os.remove(f"temp/temp_{code}_Xtrain.csv")
    os.remove(f"temp/temp_{code}_Xtest.csv")
    os.remove(f"temp/temp_{code}_ytrain.csv")
    os.remove(f"temp/temp_{code}_ytest.csv")
    
    
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train sigmoid baseline")
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
        "--mixed_effect",
        action = "store_true",
        help="Uses the mixed_effect model"
    )
    
    args= parser.parse_args()
    dataset = args.dataset
    fold = args.fold
    setting = args.setting
    config = {"env":{}}
    config["env"]["fold"] = fold
    config["env"]["dataset"] = dataset
    config["env"]["setting"] = setting
    config["env"]["mixed_effect"] = args.mixed_effect
    train_funfor(config)