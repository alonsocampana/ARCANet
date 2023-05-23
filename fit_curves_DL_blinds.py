import numpy as np
import pandas as pd
from torch import nn
import torch
from torch.utils.data import Dataset
import uuid
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, WeightedRandomSampler
from torch_geometric.data import Batch
from torch_geometric import nn as gnn
import torchmetrics
import tqdm
from functools import lru_cache
import torch_geometric
from models import ARCANet, ARCANetNoGates
from datasets import GDSC1RawDataset
from utils import JSONLogger
import argparse
from metrics import metrics_detailed, metrics_rough, kl_loss
from utils import CurveDataSplitter, CurvesPreprocessor, GDSC1RawDatasetInter


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='FitDL',
                    description='fits the DL model',)

    parser.add_argument('--setting', type=str)           # positional argument
    parser.add_argument('--fold', type=int)      # option that takes a value
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--loss', type=str)
    parser.add_argument('--drop_drugs', type=int, default = 0)
    args = parser.parse_args()
    setting = args.setting
    n_device = args.cuda
    loss = args.loss
    if setting == "precision_oncology":
        precision_oncology = True
    elif setting == "drug_discovery":
        precision_oncology = False
    dropped_drugs = args.drop_drugs
    cv = args.fold
    curves = pd.read_csv("data/matrix_corrected_GDSC2.csv")
    pp = CurvesPreprocessor(reindex = False)
    curves = pp(curves)
    if dropped_drugs is None:
        cds = CurveDataSplitter(curves,
                            smoothing = False,
                            concs_as_dilution_step = False)
        suffix = ""
    else:
        cds = CurveDataSplitter(curves,
                            smoothing = False,
                            concs_as_dilution_step = False,
                            drop_classes_training=dropped_drugs)
        suffix = f"dropped_drugs_{dropped_drugs}"
    
    data_curves = cds.get_data(curves)
    if precision_oncology:
         train_data, test_data = cds.get_split_po(data_curves, cv)
    else:
         train_data, test_data = cds.get_split_dd(data_curves, cv)
    train_ds = GDSC1RawDatasetInter(train_data["drugs"],
                              train_data["lines"],
                              train_data["inhibs"],
                              train_data["concs"])
    test_ds = GDSC1RawDatasetInter(test_data["drugs"],
                              test_data["lines"],
                              test_data["inhibs"],
                              test_data["concs"])
    config = {"optimizer": {"learning_rate": 0.0008,
                    "patience" : 4,
                    "factor": 0.5,
                    "weight_mse": 0.25,
                    "temperature" : 2.5,
                    "temperature_decay":1},
      "network": {"embed_dim" : 256,
                  "p_dropout_attn": 0.12,
                 "p_dropout_fc": 0.45,
                 "dropout_genes":0.4,
                 "dropout_nodes":0.4,
                 "n_pooling_heads":2,
                 "gat_heads":6,
                 "gat_layers":9,
                 "fc_hidden":3898,
                 "activation":3,
                 "gaussian_noise":0.05},
       "env": {"device" : f"cuda:{n_device}",
               "target_dim" : 165,
               "partition": None,
               "log_conc" : False,
               "sample_frac" : 1,
               "max_iter": 100,
               "logging": True},
     }
    root = "./logs/GDSC2/"
    if dropped_drugs != 0:
        RUN = f"{setting}_fold_{cv}_loss_{loss}_dropped_drugs_{dropped_drugs}"
    else:
        RUN = f"{setting}_fold_{cv}_loss_{loss}"
    
    if config["env"]["logging"]:
        logger = JSONLogger(root = root, path = f"{RUN}.json",
                        save_at_iter = True)
        logger(epoch = "config", config = config)
    else:
        logger = {"log":None}

    model = ARCANet(target_dim = config["env"]["target_dim"],
                       **config["network"])
    optimizer = torch.optim.Adam(model.parameters(), config["optimizer"]["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           patience = config["optimizer"]["patience"],
                                                           factor=config["optimizer"]["factor"])

    mse_loss = nn.MSELoss()
    T = config["optimizer"]["temperature"]
    T_g = config["optimizer"]["temperature_decay"]
    device = torch.device(config["env"]["device"])
    model.to(device)
    train_loader = DataLoader(train_ds,
                              batch_size=256,
                              collate_fn=Batch.from_data_list,
                              num_workers=16,
                              sampler = RandomSampler(train_ds,
                                                      replacement = True,
                                                      num_samples = int(config["env"]["sample_frac"] * len(train_ds))))
    test_loader = DataLoader(test_ds,
                            batch_size=256,
                            collate_fn=Batch.from_data_list,
                            num_workers=16,
                            shuffle=False)

    kls = []
    EXTRA_CONCS= 5
    kl = torch.nn.KLDivLoss(reduction="batchmean")
    kld = nn.KLDivLoss(reduction = "none", log_target=False)
    for epoch in range(config["env"]["max_iter"]):
        T *= T_g
        if False:
            metrics = metrics_detailed.to(device)
        else:
            metrics = metrics_rough.to(device)
        metrics.reset()
        model.train()
        kl_batch = []
        var_batch = []
        EXTRA_CONCS = 0
        for b in train_loader:
            b = b.to(device)
            y_pred = model(b).squeeze()
            y_obs = b["inhibs"].to(device).squeeze()
            if loss == "kl":
                total_kl = kl_loss(y_pred, y_obs, kl, T=T)
                total_mse = mse_loss(y_obs[:, 0], y_pred[:, 0])
                kl_batch += [total_kl.item()]
                (total_kl + config["optimizer"]["weight_mse"]*total_mse).backward()
            elif loss == "mse":
                total_mse = mse_loss(y_obs, y_pred)
                (config["optimizer"]["weight_mse"]*total_mse).backward()
            metrics(y_pred,y_obs)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step(np.nanmean(kl_batch))
        with torch.no_grad():
            m_train = {it[0]:it[1].item() for it in metrics.compute().items()}
            m_train["kl"] = np.nanmean(kl_batch)
        model.eval()
        metrics.reset()
        kl_batch = []
        EXTRA_CONCS = 0
        for b in test_loader:
            with torch.no_grad():
                b = b.to(device)
                y_pred = model(b).squeeze()
                y_obs = b["inhibs"].to(device).squeeze()
                total_mse = mse_loss(y_obs, y_pred)
                metrics(y_pred,y_obs)
        m_test = {it[0]:it[1].item() for it in metrics.compute().items()}
        best_mse = 1
        if m_test["MeanSquaredError"] < best_mse:
            best_mse = m_test["MeanSquaredError"]
            state = model.state_dict()
        if config["env"]["logging"]:
            logger(epoch = epoch, train = m_train, test = m_test)
        if (epoch + 1) % 100 == 0:              
            torch.save({"state_best": state,
                "state_last": model.state_dict(),
                "optim": optimizer.state_dict()}, f"models/GDSC2_{RUN}.pt")
