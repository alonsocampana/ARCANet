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
from utils import build_model, build_model_fingerprint, metric_dict, print_epoch, get_dataloaders, serialize_config, interpolate_tensor
import argparse
import uuid

def train_model(config, callback_intermediate = None, callback_final = None):
    if callback_intermediate is None:
        callback_intermediate = [print_epoch]
    if config["env"]["debug"]:
        debug = True
    else:
        debug = False
    train_loader, test_loader, rescale_training = get_dataloaders(config["env"]["dataset"],
                config["env"]["setting"],
                config["env"]["fold"],
                config["optimizer"]["batch_size"],
                drop_random = config["env"]["missing_random"],
                drop_systematic = config["env"]["missing_systematically"],
                leave_out = config["env"]["leave_out"])
    for x in train_loader:
        break
    def kl_loss(y_pred, y_obs, kl, eps = 0.01, mu=0.0, T = 2.5):
        y_pred_norm = F.log_softmax((y_pred.squeeze() - mu)*T, 1)
        y_obs_norm = F.softmax((y_obs.squeeze() - mu)*T, 1)
        return kl(y_pred_norm, y_obs_norm)
    scaler = torch.cuda.amp.GradScaler()
    if config["env"]["fingerprint"]:
        model = build_model_fingerprint(x, **config["network"])
    else:
        model = build_model(x, **config["network"])
    device = torch.device(f'cuda:{config["env"]["cuda_device"]}')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), config["optimizer"]["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config["optimizer"]["gamma_factor"], patience=4)
    kl = torch.nn.KLDivLoss(reduction="batchmean")
    train_metrics = torchmetrics.MetricTracker(torchmetrics.MetricCollection([torchmetrics.MeanSquaredError()])).to(device)
    test_metrics = torchmetrics.MetricTracker(torchmetrics.MetricCollection([torchmetrics.MeanSquaredError(), torchmetrics.PearsonCorrCoef()])).to(device)
    loss_fn = nn.MSELoss()
    alpha = config["optimizer"]["alpha"]
    patience_loss = 200
    for epoch in range(int(config["optimizer"]["max_iter"] * rescale_training)):
        skipped_steps = 0
        model.train()
        train_metrics.increment()
        test_metrics.increment()
        for batch in train_loader:
            batch.to(device)
            if config["env"]["interpolation_augment"] != 0:
                batch["y"] = interpolate_tensor(batch["y"], config["env"]["interpolation_augment"])
                batch["z"] = interpolate_tensor(batch["z"], config["env"]["interpolation_augment"])
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(dtype=torch.float16, enabled=config["env"]["mixed_precision"]):
                y_pred = model(batch).squeeze()
                total_loss = (1-alpha) * loss_fn(y_pred, batch["y"]) + alpha*kl_loss(y_pred, batch["y"], kl)
            if torch.isnan(total_loss):
                skipped_steps += 1
                model.reset_batchnorm()
                optimizer.zero_grad()
            else:
                scaler.scale(total_loss).backward()
                if config["optimizer"]["clip_norm"]:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["optimizer"]["clip_norm"])
                scaler.step(optimizer)
                scaler.update()
                with torch.no_grad():
                    train_metrics.update(y_pred.flatten(), batch["y"].flatten())
        if debug:
            print(f"DEBUG: skipped {skipped_steps} update steps")
        model.eval()
        for batch in test_loader:
            batch.to(device)
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.float16, enabled=config["env"]["mixed_precision"]):
                    y_pred = model(batch).squeeze()
                test_metrics.update(y_pred.flatten(), batch["y"].flatten())
        train_dict =  metric_dict(train_metrics.compute(), "_train",)
        test_dict = metric_dict(test_metrics.compute(), "_test")
        scheduler.step(train_dict["MeanSquaredError_train"])
        for fn in callback_intermediate:
            fn(epoch=epoch,train_dict=train_dict, test_dict=test_dict)
        if epoch == 0:
            best_train_loss = train_dict["MeanSquaredError_train"] + 0
        elif best_train_loss > train_dict["MeanSquaredError_train"]:
            patience_loss = 200
            best_train_loss = train_dict["MeanSquaredError_train"] + 0
        else:
            patience_loss -= 1
        if patience_loss <= 0:
            break
    if callback_final is not None:
        dataset = config["env"]["dataset"]
        setting = config["env"]["setting"]
        fusion = config["network"]["fusion"]
        fold = config["env"]["fold"]
        suffix =  f"{dataset}_{setting}_{fusion}"
        if config["env"]["fingerprint"]:
            suffix += "_fingerprint"
        if config["optimizer"]["alpha"] == 0:
            suffix += "_nokl"
        if config["network"]["linear_head"]:
            suffix += "_linearhead"
        if config["env"]["missing_random"]:
            suffix += f'_random_{config["env"]["missing_random"]}'
        elif config["env"]["missing_systematically"]:
            suffix += f'_systematic_{config["env"]["missing_systematically"]}'
        suffix += f"_{fold}"
        try:
            if config["env"]["random_suffix"]:
                suffix += f"_{str(uuid.uuid4())}"
        except KeyError:
            pass
        callback_final(model, train_metrics, test_metrics, suffix)
    return test_dict["PearsonCorrCoef_test"]
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameters study")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="The dataset you want to use (a string)."
    )
    parser.add_argument(
        "--dataset_hyperpar",
        type=str,
        required=True,
        help="The dataset used for hyperparameter optimization."
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
        "--setting_hyperpar",
        type=str,
        required=True,
        help="The dataset used during hyperparameter optimization"
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
        "--fusion_concat",
        action = "store_true",
        help="Use fusion concatenation instead of Hill",
    )
    
    parser.add_argument(
        "--nokl",
        action = "store_true",
        help="Use only MSE for training",
    )
    parser.add_argument(
        "--linear_head",
        action = "store_true",
        help="Replace MLP head by linear",
    )
    parser.add_argument(
        "--random_suffix",
        action = "store_true",
        help="Adds random suffix to name",
    )
    
    # parser.add_argument(
    #     "--log_conc",
    #     action = "store_true",
    #     help="Log transform the concentrations",
    # )
    parser.add_argument(
        "--interpolation_augment",
        type=int,
        required=False,
        default = 0,
        help="Augment training data by performing linear interpolation"
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
    def save_model_and_results(model, train_metrics, test_metrics, suffix):
        torch.save(model.state_dict(), f"models/{suffix}.pt")
        dict_test = {it[0]+"_test":it[1].detach().cpu().numpy() for it in test_metrics.compute_all().items()}
        dict_train = {it[0]+"_train":it[1].detach().cpu().numpy() for it in train_metrics.compute_all().items()}
        metrics_df = pd.DataFrame({**dict_test, **dict_train})
        metrics_df.to_csv(f"results/{suffix}.csv")

    args= parser.parse_args()
    fingerprint = args.fingerprint
    fusion_concat = args.fusion_concat
    dataset = args.dataset
    fold = args.fold
    setting = args.setting
    cuda_device = args.cuda
    max_iter = args.max_iter
    linear_head = args.linear_head
    config = serialize_config(args.dataset_hyperpar, args.setting_hyperpar)
    config["env"]["fold"] = fold
    config["env"]["dataset"] = dataset
    config["env"]["setting"] = setting
    config["optimizer"]["max_iter"] = max_iter
    config["env"]["cuda_device"] = cuda_device
    config["env"]["fingerprint"] = fingerprint
    config["env"]["missing_systematically"] = args.missing_systematically
    config["env"]["missing_random"] = args.missing_random
    config["env"]["interpolation_augment"] = args.interpolation_augment
    config["env"]["leave_out"] = None
    config["env"]["random_suffix"] = args.random_suffix
    config["network"]["linear_head"] = linear_head
    # if args.log_conc:
    #     config["network"]["transform_log_conc"] = True
    if fusion_concat:
        config["network"]["fusion"] = "concat"
    if args.nokl:
        config["optimizer"]["alpha"] = 0
    train_model(config, callback_final = save_model_and_results)