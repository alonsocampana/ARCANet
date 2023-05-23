import numpy as np
import pandas as pd
from torch import nn
import torch
from torch.utils.data import Dataset
import uuid
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
from models import SigmoidModel
from datasets import GDSC1RawDataset
from utils import JSONLogger
import argparse
from utils import CurvesPreprocessor, CurveDataSplitter
from metrics import metrics_detailed, metrics_rough, kl_loss
from utils import CurveDataSplitter, CurvesPreprocessor, GDSC1RawDatasetInter

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='FitDL',
                    description='fits the DL model',)

    parser.add_argument('--setting', type=str)           # positional argument
    parser.add_argument('--dropped', type=int)
    parser.add_argument('--cuda', type=int)
    parser.add_argument('--parameters', type=int)
    parser.add_argument('--drop_random', type=int, default = 0)
    parser.add_argument('--drop_classes', type=int, default = 0)
    args = parser.parse_args()
    setting = args.setting
    n_device = args.cuda
    parameters = args.parameters
    if setting == "smoothing":
        smoothing = True
        max_fold = 7
    elif setting == "interpolation":
        smoothing = False
        max_fold = 5
    dropped = args.dropped
    curves = pd.read_csv("data/matrix_corrected_GDSC2.csv")
    pp = CurvesPreprocessor(reindex = True)
    curves = pp(curves)
    cds = CurveDataSplitter(curves,
                            smoothing = smoothing,
                            concs_as_dilution_step = False)
    data_curves = cds.get_data(curves)
    if parameters == 4:
        n_par = "4P"
    elif parameters == 3:
        n_par = "3P"
    root = "./logs/GDSC2/"
    RUN = f"sigmoid{n_par}_{setting}_dropped_{dropped}"
    cvmse = {}
    cv = 0
    device = torch.device(f"cuda:{n_device}")
    while cv < max_fold:
        max_retries = 5
        if dropped == 1:
             train_concs, train_inhibs, test_concs, test_inhibs = cds.get_split(data_curves, cv)
        else:
            if setting == "smoothing":
                ks = cds.get_multiplek(dropped, n_cv=7)
            else:
                ks = cds.get_multiplek(dropped, n_cv=5)
            _, _, test_concs, test_inhibs = cds.get_split(data_curves, cv)
            train_concs, train_inhibs, _, _ = cds.get_split_multiplek(data_curves, list(ks[:, cv]))
        sig = SigmoidModel(len(pp.unique_drugs), len(pp.unique_lines), device, parameters = parameters)
        sig.to(device)
        mse = nn.MSELoss()
        optim = torch.optim.AdamW(sig.parameters(), 0.1, weight_decay = 0.00001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor= 0.75, patience = 50)
        patience = 100
        last_l2 = [1]
        retry = False
        for epoch in range(30000):
            sig.train()
            pred = sig(torch.Tensor(data_curves["drugs"]).long().to(device),
                       torch.Tensor(data_curves["lines"]).long().to(device),
                       torch.Tensor(train_concs).to(device))
            l = mse(pred, torch.clip(torch.Tensor(train_inhibs).to(device), 0, 1))
            l.backward()
            optim.step()
            optim.zero_grad()
            scheduler.step(l.item())
            sig.eval()
            with torch.no_grad():
                pred = sig(torch.Tensor(data_curves["drugs"]).long().to(device),
                       torch.Tensor(data_curves["lines"]).long().to(device),
                       torch.Tensor(test_concs).to(device))
                l2 = mse(pred.squeeze(), torch.Tensor(test_inhibs).to(device).squeeze())
            if last_l2[0] > l2:
                patience = 100
            else:
                patience -= 1
            if patience <= 0:
                if epoch <= 1000:
                    max_retries -= 1
                    if max_retries > 0:
                        retry = True
                break
            last_l2 = [l2]
        cvmse[cv] = l2
        if not retry:
            cv += 1
        print(f"epoch: {epoch}, loss: {l.item()}, test_loss {l2.item()}")
    pd.DataFrame([it.item() for it in list(cvmse.values())]).assign(dropped = dropped).to_csv(f"results/baseline/{RUN}.csv")
