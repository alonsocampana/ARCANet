import pandas as pd
import torch 
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, WeightedRandomSampler
from torch_geometric.data import Batch
import numpy as np
from torch_geometric import nn as gnn
import torchmetrics
import tqdm
from functools import lru_cache
import torch_geometric
import uuid
from models import ARCANEt
from datasets import GDSC1RawDataset
from utils import JSONLogger
import argparse
import optuna

def train_model(trial, config, RUN, root = f"logs/GDSC1/"):
    if config["env"]["logging"]:
        logger = JSONLogger(root = root, path = f"{RUN}.json",
                        save_at_iter = True)
        logger(epoch = "config", config = config)
    else:
        logger = {"log":None}
    config["optimizer"] = {"learning_rate": trial.suggest_float("learning_rate", 0.00000001, 0.1, log=True),
                    "patience" : 4,
                    "factor": 0.5,
                    "weight_mse": trial.suggest_float("weight_mse", 0.00, 1),
                    "temperature": trial.suggest_float("temperature", 0.1, 20),
                    "temperature_decay": trial.suggest_float("temperature_decay", 0.5, 2)}
    
    config["network"] = {"embed_dim" : trial.suggest_int("embed_dim", 64, 1024),
                  "p_dropout_attn":trial.suggest_float("p_dropout_attn", 0.0, 0.50),
                 "p_dropout_fc":trial.suggest_float("p_dropout_fc", 0.0, 0.50),
                 "dropout_genes":trial.suggest_float("dropout_genes", 0.0, 0.50),
                 "dropout_nodes":trial.suggest_float("dropout_nodes", 0.0, 0.50),
                 "n_pooling_heads":trial.suggest_int("n_pooling_heads", 1, 6),
                 "gat_heads":trial.suggest_int("gat_heads", 1, 6),
                 "gat_layers":trial.suggest_int("gat_layers", 1, 10),
                 "fc_hidden":trial.suggest_int("fc_hidden", ()),
                 "activation":trial.suggest_int("activation", 0, 4),
                 "gaussian_noise":trial.suggest_float("gaussian_noise", 0, 0.1)}
    model = MultiModel(target_dim = config["env"]["target_dim"],
                       **config["network"])
    optimizer = torch.optim.Adam(model.parameters(), config["optimizer"]["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           patience = config["optimizer"]["patience"],
                                                           factor=config["optimizer"]["factor"])
    
    mse_loss = nn.MSELoss()
    device = torch.device(config["env"]["device"])
    model.to(device)

    data = pd.read_csv("data/matrix_corrected_GDSC1.csv")

    data = data[(~data.iloc[:, 2:].isna()).sum(axis=1) <=9]

    col = config["env"]["partition"]
    T = config["optimizer"]["temperature"]
    T_g = config["optimizer"]["temperature_decay"]
    if col != "none":
        np.random.seed(3558)
        folds = np.array_split(np.random.permutation(data[f"{col}"].unique()), 10)
        test_fold = folds[0]
        val_fold = folds[1]
        train_ds = GDSC1RawDataset(data.query(f"({col} not in @val_fold) & ({col} not in @test_fold)"), add_training_concs=0, log_conc=config["env"]["log_conc"])
        val_ds = GDSC1RawDataset(data.query(f"({col} in @val_fold)"))
    else:
        np.random.seed(3558)
        index = np.arange(len(data))
        folds = np.array_split(np.random.permutation(index), 10)
        test_fold = folds[0]
        val_fold = folds[1]
        train_fold = np.concatenate([folds[i] for i in range(2, len(folds))])
        train_ds = GDSC1RawDataset(data.iloc[train_fold], add_training_concs=0, log_conc=config["env"]["log_conc"])
        val_ds = GDSC1RawDataset(data.iloc[val_fold])
    train_loader = DataLoader(train_ds,
                              batch_size=256,
                              collate_fn=Batch.from_data_list,
                              num_workers=16,
                              sampler = RandomSampler(train_ds,
                                                      replacement = True,
                                                      num_samples = int(config["env"]["sample_frac"] * len(train_ds))))
    val_loader = DataLoader(val_ds,
                            batch_size=2096,
                            collate_fn=Batch.from_data_list,
                            num_workers=16,
                            shuffle=False)

    kls = []

    from metrics import metrics_detailed, metrics_rough, kl_loss

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
            y_pred = model(b)
            y_pred_f = y_pred.squeeze()[~b["y"].isnan()]
            y_obs_f = b["y"].to(device).squeeze()[~b["y"].isnan()]
            nan_target = b["y"].isnan()
            y_5 = y_pred[(~nan_target).sum(axis=1) == 5+EXTRA_CONCS].squeeze()
            y_9 = y_pred[(~nan_target).sum(axis=1) == 9+EXTRA_CONCS].squeeze()
            b_5 = b["y"][(~nan_target).sum(axis=1) == 5+EXTRA_CONCS].squeeze()
            b_9 = b["y"][(~nan_target).sum(axis=1) == 9+EXTRA_CONCS].squeeze()
            y_obs_5 = b_5[~b_5.isnan()].reshape(-1, 5+EXTRA_CONCS)
            y_pred_5 = y_5[~b_5.isnan()].reshape(-1, 5+EXTRA_CONCS)
            y_obs_9 = b_9[~b_9.isnan()].reshape(-1, 9+EXTRA_CONCS)
            y_pred_9 = y_9[~b_9.isnan()].reshape(-1, 9+EXTRA_CONCS)
            frac_5 = y_obs_5.shape[0]/ (y_obs_5.shape[0] + y_obs_9.shape[0])
            total_kl = frac_5 * kl_loss(y_pred_5, y_obs_5, kl, T=T) + (1 - frac_5) * kl_loss(y_pred_9, y_obs_9, kl, T=T)
            total_mse = frac_5 * mse_loss(y_obs_5[:, 0], y_pred_5[:, 0]) + (1 - frac_5) * mse_loss(y_obs_9[:, 0], y_pred_9[:, 0])
            metrics(y_pred_f,y_obs_f)
            kl_batch += [total_kl.item()]
            (total_kl + config["optimizer"]["weight_mse"]*total_mse).backward()
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
        y_obs_5_ = []
        y_pred_5_ = []
        y_obs_9_ = []
        y_pred_9_ = []
        for b in val_loader:
            with torch.no_grad():
                b = b.to(device)
                y_pred = model(b)
                y_pred_f = y_pred.squeeze()[~b["y"].isnan()]
                y_obs_f = b["y"].to(device).squeeze()[~b["y"].isnan()]
                nan_target = b["y"].isnan()
                y_5 = y_pred[(~nan_target).sum(axis=1) == 5+EXTRA_CONCS].squeeze()
                y_9 = y_pred[(~nan_target).sum(axis=1) == 9+EXTRA_CONCS].squeeze()
                b_5 = b["y"][(~nan_target).sum(axis=1) == 5+EXTRA_CONCS].squeeze()
                b_9 = b["y"][(~nan_target).sum(axis=1) == 9+EXTRA_CONCS].squeeze()
                y_obs_5 = b_5[~b_5.isnan()].reshape(-1, 5+EXTRA_CONCS)
                y_pred_5 = y_5[~b_5.isnan()].reshape(-1, 5+EXTRA_CONCS)
                y_obs_9 = b_9[~b_9.isnan()].reshape(-1, 9+EXTRA_CONCS)
                y_pred_9 = y_9[~b_9.isnan()].reshape(-1, 9+EXTRA_CONCS)
                y_obs_5_ += [y_obs_5]
                y_pred_5_ += [y_pred_5]
                y_obs_9_ += [y_obs_9]
                y_pred_9_ += [y_pred_9]
                frac_5 = y_obs_5.shape[0]/ (y_obs_5.shape[0] + y_obs_9.shape[0])
                total_kl = frac_5 * kl_loss(y_pred_5, y_obs_5, kl) + (1 - frac_5) * kl_loss(y_pred_9, y_obs_9, kl)
                total_mse = frac_5 * mse_loss(y_obs_5[:, 0], y_pred_5[:, 0]) + (1 - frac_5) * mse_loss(y_obs_9[:, 0], y_pred_9[:, 0])
                metrics(y_pred_f,y_obs_f)
                kl_batch += [total_kl.item()]
        m_test = {it[0]:it[1].item() for it in metrics.compute().items()}
        m_test["kl"] = np.nanmean(kl_batch)
        y_obs_5 = torch.cat(y_obs_5_)
        y_obs_9 = torch.cat(y_obs_9_)
        y_pred_5 = torch.cat(y_pred_5_)
        y_pred_9 = torch.cat(y_pred_9_)
        val_9 = val_loader.dataset.data.loc[val_loader.dataset.data.isna().sum(axis=1) == 156]
        val_5 = val_loader.dataset.data.loc[val_loader.dataset.data.isna().sum(axis=1) == 160]
        val_5 = val_5.assign(var_obs = y_obs_5.var(axis=1).cpu().numpy(), var_pred = y_pred_5.var(axis=1).cpu().numpy())
        val_9 = val_9.assign(var_obs = y_obs_9.var(axis=1).cpu().numpy(), var_pred = y_pred_9.var(axis=1).cpu().numpy())
        val_df = pd.concat([val_5, val_9])
        mean_corr = val_df.groupby("DRUG_ID")["var_obs", "var_pred"].corr().iloc[0::2,-1].mean()
        trial.report(mean_corr, step = epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
        best_kl = 1
        if m_test["kl"] < best_kl:
            best_kl = m_test["kl"]
            state = model.state_dict()
        if config["env"]["logging"]:
            logger(epoch = epoch, train = m_train, test = m_test)
        if (epoch + 1) % 100 == 0:              
            torch.save({"state_best": state,
                "state_last": model.state_dict(),
                "optim": optimizer.state_dict()}, f"models/GDSC1_{RUN}.pt")
    return mean_corr

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--cuda",
        type=int,
        required=False,
        default=0,
        help="Selects cuda device")
    parser.add_argument(
        "--fraction",
        type=float,
        required=False,
        default=1,
        help="Data fraction used at each training iteration")
    parser.add_argument(
        "--cold_start",
        type=str,
        required=False,
        default="none",
        help="One of none, COSMIC_ID or DRUG_ID")
    args, _ = parser.parse_known_args()
    config = {"optimizer": {"learning_rate": 0.0001,
                    "patience" : 4,
                    "factor": 0.5,
                    "weight_mse": 0.01,
                    "temperature" : 5,
                    "temperature_decay":1},
      "network": {"embed_dim" : 512,
                  "p_dropout_attn":0.1,
                 "p_dropout_fc":0.2,
                 "dropout_genes":0.4,
                 "dropout_nodes":0.3,
                 "n_pooling_heads":2,
                 "gat_heads":2,
                 "gat_layers":8,
                 "fc_hidden":2048,
                 "activation":0,
                 "gaussian_noise":0},
       "env": {"device" : f"cuda:{args.cuda}",
               "target_dim" : 165,
               "partition": args.cold_start,
               "log_conc" : False,
               "sample_frac" : args.fraction,
               "max_iter": 200,
               "logging": False},
     }
    RUN = str(uuid.uuid4())
    objective = lambda x: train_model(x, config, RUN)
    study_name = "cf_prec_onc_t2"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(study_name=study_name,
                                storage=storage_name,
                                direction='maximize',
                                load_if_exists=True,
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=2,
                                                               n_warmup_steps=10,
                                                               interval_steps=10))
    study.optimize(objective, n_trials=50)