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
    parser.add_argument('--dropped', type=int)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--loss', type=str)
    parser.add_argument('--embed_size', type=int, default = 256)
    parser.add_argument('--gat_layers', type=int, default = 9)
    parser.add_argument('--gat_heads', type=int, default = 6)
    parser.add_argument('--temperature', type=float, default = 2.5)
    parser.add_argument('--weight_mse', type=float, default = 0.25)
    parser.add_argument('--gaussian_noise', type=float, default = 0.05)
    parser.add_argument('--activation', type=int, default = 3)
    parser.add_argument('--learn_gates', type=bool, default = True)
    parser.add_argument('--no_gates',action='store_true')
    args = parser.parse_args()
    setting = args.setting
    n_device = args.cuda
    loss = args.loss
    #ablation study terms
    embed_size = args.embed_size
    gat_layers = args.gat_layers
    gat_heads = args.gat_heads
    temperature = args.temperature
    weight_mse = args.weight_mse
    gaussian_noise = args.gaussian_noise
    activation = args.activation
    learn_gates = not args.no_gates
    if setting == "smoothing":
        smoothing = True
    elif setting == "interpolation":
        smoothing = False
    cv = args.fold
    dropped = args.dropped
    curves = pd.read_csv("data/matrix_corrected_GDSC2.csv")
    pp = CurvesPreprocessor(reindex = False)
    curves = pp(curves)
    cds = CurveDataSplitter(curves,
                            smoothing = smoothing,
                            concs_as_dilution_step = False)
    data_curves = cds.get_data(curves)
    if dropped == 1:
         train_concs, train_inhibs, test_concs, test_inhibs = cds.get_split(data_curves, cv)
    else:
        if setting == "smoothing":
            ks = cds.get_multiplek(dropped, n_cv=7)
        else:
            ks = cds.get_multiplek(dropped, n_cv=5)
        _, _, test_concs, test_inhibs = cds.get_split(data_curves, cv)
        train_concs, train_inhibs, _, _ = cds.get_split_multiplek(data_curves, list(ks[:, cv]))
    train_ds = GDSC1RawDatasetInter(data_curves["drugs"],
                              data_curves["lines"],
                              train_inhibs,
                              train_concs)
    test_ds = GDSC1RawDatasetInter(data_curves["drugs"],
                              data_curves["lines"],
                              test_inhibs,
                              test_concs)
    config = {"optimizer": {"learning_rate": 0.0008,
                    "patience" : 4,
                    "factor": 0.5,
                    "weight_mse": weight_mse,
                    "temperature" : temperature,
                    "temperature_decay":1},
      "network": {"embed_dim" : embed_size,
                  "p_dropout_attn": 0.12,
                 "p_dropout_fc": 0.45,
                 "dropout_genes":0.4,
                 "dropout_nodes":0.4,
                 "n_pooling_heads":2,
                 "gat_heads":gat_heads,
                 "gat_layers":gat_layers,
                 "fc_hidden":3898,
                 "activation":activation,
                 "gaussian_noise":gaussian_noise},
       "env": {"device" : f"cuda:{n_device}",
               "target_dim" : 165,
               "partition": None,
               "log_conc" : False,
               "sample_frac" : 1,
               "max_iter": 100,
               "logging": True},
     }
    root = "./logs/GDSC2/ablation/"
    RUN = f"{setting}_fold_{cv}_loss_{loss}_{embed_size}_{gat_layers}_{gat_heads}_{temperature}_{weight_mse}_{gaussian_noise}_{activation}_{learn_gates}"
    if config["env"]["logging"]:
        logger = JSONLogger(root = root, path = f"{RUN}.json",
                        save_at_iter = True)
        logger(epoch = "config", config = config)
    else:
        logger = {"log":None}
    if learn_gates:
        model = ARCANet(target_dim = config["env"]["target_dim"],
                       **config["network"])
    else:
        model = ARCANetNoGates(target_dim = config["env"]["target_dim"],
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
