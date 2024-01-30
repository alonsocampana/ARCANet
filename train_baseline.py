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
from utils import build_model, build_model_fingerprint, metric_dict, print_epoch, get_train_test_data, serialize_config
import argparse
import uuid
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

def as_mapped_dict(arry):
    max_ar = arry.max()
    min_ar = arry.min()
    transformed_cs = np.log2(arry/min_ar) + 1
    return {arry[i]:transformed_cs[i] for i in range(len(arry))}

def train_baseline(config):
    train_data, test_data = get_train_test_data(config["env"]["dataset"],
                config["env"]["setting"],
                config["env"]["fold"])
    all_data = pd.concat([train_data, test_data])
    map_cells = {cell:idx+1 for idx, cell in enumerate(all_data.loc[:, "cell"].unique())}
    map_drugs = {drug:idx+1 for idx, drug in enumerate(all_data.loc[:, "drug"].unique())}
    train_data.loc[:, "cell"] = train_data.loc[:, "cell"].map(map_cells)
    test_data.loc[:, "cell"] = test_data.loc[:, "cell"].map(map_cells)
    all_data.loc[:, "cell"] = all_data.loc[:, "cell"].map(map_drugs)
    train_data.loc[:, "drug"] = train_data.loc[:, "drug"].map(map_drugs)
    test_data.loc[:, "drug"] = test_data.loc[:, "drug"].map(map_drugs)
    all_data.loc[:, "drug"] = all_data.loc[:, "drug"].map(map_drugs)
    max_cs = all_data.groupby("drug")["z"].max()
    dic_cs = all_data.groupby("drug")["z"].unique().to_dict()
    map_drug_cs = {it[0]:as_mapped_dict(it[1]) for it in dic_cs.items()}
    train_data.loc[:, "z"] = train_data.apply(lambda x: map_drug_cs[x["drug"]][x["z"]], 1)
    test_data.loc[:, "z"] = test_data.apply(lambda x: map_drug_cs[x["drug"]][x["z"]], 1)
    rename_dict = {"cell": "CL", "z":"x"}
    train_data = train_data.rename(columns=rename_dict)
    test_data = test_data.rename(columns=rename_dict)
    train_data = train_data.assign(maxc = max_cs.loc[train_data.loc[:, "drug"]].to_numpy())
    test_data = test_data.assign(maxc = max_cs.loc[test_data.loc[:, "drug"]].to_numpy())
    train_data["y"] = 1-train_data["y"]
    test_data["y"] = 1-test_data["y"]
    code = str(uuid.uuid4())
    train_data = train_data.loc[:,["drug", "CL", "x", "y", "maxc"]]
    test_data = test_data.loc[:,["drug", "CL", "x", "y", "maxc"]]
    train_data.to_csv(f"temp/temp_{code}_train.csv")
    test_data.to_csv(f"temp/temp_{code}_test.csv")
    if config["env"]["mixed_effect"]:
        command = f"Rscript sigmoid2P.R {code}"
        os.system(command)
        y_pred = pd.read_csv(f"temp/temp_{code}_pred.csv").to_numpy().squeeze()
        y_obs = test_data.loc[:, "y"].to_numpy()
        pd.Series({"MeanSquaredError_test" : mean_squared_error(y_obs, y_pred), "PearsonCorrCoef_test" : pearsonr(y_obs, y_pred)[0]}).to_csv(f"results_baseline/{config['env']['dataset']}_{config['env']['setting']}_{config['env']['fold']}.csv")
        os.remove(f"temp/temp_{code}_train.csv")
        os.remove(f"temp/temp_{code}_test.csv")
        os.remove(f"temp/temp_{code}_pred.csv")
    else:
        command = f"Rscript Rsigmoids.R {code}"
        os.system(command)
        pred_df = pd.read_csv(f"temp/temp_{code}_pred.csv")
        pred_df = test_data.merge(pred_df, how="left").fillna(0)
        print(pred_df.columns)
        y_pred = pred_df.loc[:, "prediction"].to_numpy()
        y_obs = test_data.loc[:, "y"].to_numpy()
        pd.Series({"MeanSquaredError_test" : mean_squared_error(y_obs, y_pred), "PearsonCorrCoef_test" : pearsonr(y_obs, y_pred)[0]}).to_csv(f"results_baseline/2P_{config['env']['dataset']}_{config['env']['setting']}_{config['env']['fold']}.csv")
        y_pred = pred_df.loc[:, "prediction.1"].to_numpy()
        y_obs = test_data.loc[:, "y"].to_numpy()
        pd.Series({"MeanSquaredError_test" : mean_squared_error(y_obs, y_pred), "PearsonCorrCoef_test" : pearsonr(y_obs, y_pred)[0]}).to_csv(f"results_baseline/3P_{config['env']['dataset']}_{config['env']['setting']}_{config['env']['fold']}.csv")
        y_pred = pred_df.loc[:, "prediction.2"].to_numpy()
        y_obs = test_data.loc[:, "y"].to_numpy()
        pd.Series({"MeanSquaredError_test" : mean_squared_error(y_obs, y_pred), "PearsonCorrCoef_test" : pearsonr(y_obs, y_pred)[0]}).to_csv(f"results_baseline/4P_{config['env']['dataset']}_{config['env']['setting']}_{config['env']['fold']}.csv")
        y_pred = pred_df.loc[:, "prediction.3"].to_numpy()
        y_obs = test_data.loc[:, "y"].to_numpy()
        pd.Series({"MeanSquaredError_test" : mean_squared_error(y_obs, y_pred), "PearsonCorrCoef_test" : pearsonr(y_obs, y_pred)[0]}).to_csv(f"results_baseline/5P_{config['env']['dataset']}_{config['env']['setting']}_{config['env']['fold']}.csv")
        
    
    
    
    
    
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
    train_baseline(config)