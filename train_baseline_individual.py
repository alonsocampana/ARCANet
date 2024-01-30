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
import torch
from torch import nn
import numpy as np
import pandas as pd
from utils import get_train_test_data
import torchmetrics

class EarlyStop():
    def __init__(self, max_patience, maximize=False):
        self.maximize=maximize
        self.max_patience = max_patience
        self.best_loss = None
        self.patience = max_patience + 0
    def __call__(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
            self.patience = self.max_patience + 0
        elif loss < self.best_loss:
            self.best_loss = loss
            self.patience = self.max_patience + 0
        else:
            self.patience -= 1
        return not bool(self.patience)
        
class LL4(nn.Module):
    def __init__(self, n_drugs, n_cells, device, P3 = False, P4 = True):
        super().__init__()
        self.b = nn.Parameter(torch.randn([n_drugs, n_cells])*0.001)
        self.e = nn.Parameter(torch.randn([n_drugs, n_cells])*0.001)
        if P3:
            self.c = nn.Parameter(torch.ones([n_drugs, n_cells]) * 0.0001)
        else:
            self.c = torch.zeros([n_drugs, n_cells]).to(device)
        if P4:
            self.d = nn.Parameter(torch.ones([n_drugs, n_cells]) * 0.0001)
        else:
            self.d = torch.ones([n_drugs, n_cells]).to(device)
        self.to(device)
    def forward(self, x, drug_id, cell_id):
        b = self.b[drug_id, cell_id]
        e = self.e[drug_id, cell_id]
        c = self.c[drug_id, cell_id]
        d = self.d[drug_id, cell_id]
        return c + (d-c)*torch.sigmoid(b*(x + e))
    
class LL4Mixed(nn.Module):
    def __init__(self, n_drugs, n_cells, device, P3 = False, P4 = True, var_samples_fixed = 0.0001, var_samples_mixed=0.0001):
        super().__init__()
        self.b = nn.Parameter(torch.randn([n_drugs, n_cells])*var_samples_mixed)
        self.b_l = nn.Parameter(torch.randn([n_drugs])*var_samples_fixed)
        self.e = nn.Parameter(torch.randn([n_drugs, n_cells])*var_samples_mixed)
        self.e_l = nn.Parameter(torch.randn([n_drugs])*var_samples_fixed)
        if P3:
            self.c = nn.Parameter(torch.ones([n_drugs, n_cells]) * var_samples_mixed)
            self.c_l = nn.Parameter(torch.ones([n_drugs]) * var_samples_fixed)
        else:
            self.c = torch.zeros([n_drugs, n_cells]).to(device)
            self.c_l = torch.zeros([n_drugs]).to(device)
        if P4:
            self.d = nn.Parameter(torch.ones([n_drugs, n_cells]) * var_samples_mixed)
            self.d_l = nn.Parameter(torch.ones([n_drugs]) * var_samples_fixed)
        else:
            self.d = torch.zeros([n_drugs, n_cells]).to(device)
            self.d_l = torch.ones([n_drugs]).to(device)
        self.to(device)
    def forward(self, x, drug_id, cell_id):
        b = self.b[drug_id, cell_id] + self.b_l[drug_id]
        e = self.e[drug_id, cell_id] + self.e_l[drug_id]
        c = self.c[drug_id, cell_id] + self.c_l[drug_id]
        d = self.d[drug_id, cell_id] + self.d_l[drug_id]
        return c + (d-c)*torch.sigmoid(b*(x + e))

def as_mapped_dict(arry):
    max_ar = arry.max()
    min_ar = arry.min()
    transformed_cs = np.log2(arry/min_ar) + 1
    return {arry[i]:transformed_cs[i] for i in range(len(arry))}
def prepare_df(df):
    df_ = df.copy()
    df_["drug"]-= 1
    df_["CL"]-= 1
    df_["x"]+= np.log2(df_["maxc"])
    return df_
def get_tensors(df, device):
    CL = torch.Tensor(df["CL"].to_numpy()).to(device).long()
    drug = torch.Tensor(df["drug"].to_numpy()).to(device).long()
    x = torch.Tensor(df["x"].to_numpy()).to(device)
    y = torch.Tensor(df["y"].to_numpy()).to(device)
    return x, y, CL, drug
def train_baseline(config):
    train_data, test_data, _, _ = get_train_test_data(config["env"]["dataset"],
                config["env"]["setting"],
                config["env"]["fold"],
                config["env"]["missing_random"],
                config["env"]["missing_systematically"])
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
    train = train_data.loc[:,["drug", "CL", "x", "y", "maxc"]]
    test = test_data.loc[:,["drug", "CL", "x", "y", "maxc"]]
    device = torch.device(f'cuda:{config["env"]["n_device"]}')
    train = prepare_df(train)
    n_cells = train["CL"].max() + 1
    n_drugs = train["drug"].max() + 1
    x_train, y_train, CL_train, drug_train = get_tensors(train, device)
    test = prepare_df(test)
    x_test, y_test, CL_test, drug_test = get_tensors(test, device)    
    model = config["env"]["model"]
    if config["env"]["mixed_effect"]:
        model_class = LL4Mixed
    else:
        model_class = LL4
    if model == "2P":
        model = model_class(n_drugs, n_cells, device, False, False)
    elif model == "3P":
         model = model_class(n_drugs, n_cells, device, False, True)
    elif model == "4P":
         model = model_class(n_drugs, n_cells, device, True, True)
    optim = torch.optim.AdamW(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=50, factor=0.1)
    loss = nn.MSELoss()
    early_stop = EarlyStop(max_patience = 10)
    for epoch in range(50000):
        optim.zero_grad()
        y_pred = model(x_train, drug_train, CL_train)
        l = loss(y_pred, y_train)
        l.backward()
        optim.step()
        scheduler.step(l.item())
        if (epoch+1) % 10 == 1:
            with torch.no_grad():
                y_pred = model(x_test, drug_test, CL_test)
                r = torchmetrics.functional.pearson_corrcoef(y_pred, y_test)
                m = torchmetrics.functional.mean_squared_error(y_pred, y_test)
            if early_stop(m):
                break
        if config["env"]["mixed_effect"]:
            suffix = f"{config['env']['model']}_{config['env']['dataset']}_{config['env']['setting']}_mixed_effect_{config['env']['fold']}"
        else:
            suffix = f"{config['env']['model']}_{config['env']['dataset']}_{config['env']['setting']}_{config['env']['fold']}"
        if config["env"]["missing_random"]:
            suffix += f'_random_{config["env"]["missing_random"]}'
        elif config["env"]["missing_systematically"]:
            suffix += f'_systematic_{config["env"]["missing_systematically"]}'
        pd.Series({"MeanSquaredError_test" : m.item(), "PearsonCorrCoef_test" : r.item()}).to_csv(f"results_baseline/{suffix}.csv")
    return r
    
    
    
    
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
        "--cuda",
        type=int,
        required=True,
        help="The cuda device number"
    )
    parser.add_argument(
        "--setting",
        type=str,
        required=True,
        help="The partition strategy you want to use."
    )
    parser.add_argument(
        "--logistic",
        type=str,
        required=True,
        help="The number of parameters employed by the sigmoid, either 2P, 3P or 4P"
    )
    parser.add_argument(
        "--mixed_effect",
        action = "store_true",
        help="Applies mixed effect"
    )
    parser.add_argument(
        "--missing_random",
        type=float,
        required=False,
        default = 0.0,
        help="Reduce the amount of data randomly"
    )
    parser.add_argument(
        "--missing_systematically",
        type=float,
        required=False,
        default = 0.0,
        help="Reduce the amount of data systematically"
    )
    args= parser.parse_args()
    dataset = args.dataset
    fold = args.fold
    setting = args.setting
    config = {"env":{"n_device":args.cuda}}
    config["env"]["fold"] = fold
    config["env"]["dataset"] = dataset
    config["env"]["setting"] = setting
    config["env"]["model"] = args.logistic
    config["env"]["mixed_effect"] = args.mixed_effect
    config["env"]["missing_systematically"] = args.missing_systematically
    config["env"]["missing_random"] = args.missing_random
    train_baseline(config)