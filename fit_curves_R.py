import numpy as np
import pandas as pd
import uuid
import tqdm
from functools import lru_cache
import uuid
from models import SigmoidModel
from datasets import GDSC1RawDataset
from utils import JSONLogger
import argparse
from utils import CurvesPreprocessor, CurveDataSplitter
from metrics import metrics_detailed, metrics_rough, kl_loss
from utils import CurveDataSplitter, CurvesPreprocessor, GDSC1RawDatasetInter, run_2P_baselines_smoothing

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='FitDL',
                    description='fits the DL model',)

    parser.add_argument('--setting', type=str)           # positional argument
    parser.add_argument('--dropped', type=int)
    parser.add_argument('--drop_random', type=int, default = 0)
    parser.add_argument('--drop_classes', type=int, default = 0)
    args = parser.parse_args()
    setting = args.setting
    drop_random = args.drop_random
    drop_classes = args.drop_classes
    suffix = ""
    if setting == "smoothing":
        smoothing = True
        max_fold = 7
    elif setting == "interpolation":
        smoothing = False
        max_fold = 5
    dropped = args.dropped
    curves = pd.read_csv("data/matrix_corrected_GDSC2.csv")
    if drop_classes != 0:
        pp = CurvesPreprocessor(reindex = False, drop_data_ids_10x=drop_classes)
        suffix += f"_classes_{drop_classes}"
    elif drop_random != 0:
        pp = CurvesPreprocessor(reindex = False, drop_at_random_10x=drop_random)
        suffix += f"_random_{drop_random}"
    else:
        pp = CurvesPreprocessor(reindex = False)
    curves = pp(curves)
    cds = CurveDataSplitter(curves,
                            smoothing = smoothing,
                            concs_as_dilution_step = True)
    data_curves = cds.get_data(curves)
    n_par = "2P"
    perfs = []
    for cv in range(max_fold):
        if dropped == 1:
             train_concs, train_inhibs, test_concs, test_inhibs = cds.get_split(data_curves, cv)
        else:
            if setting == "smoothing":
                ks = cds.get_multiplek(dropped, n_cv=7)
            else:
                ks = cds.get_multiplek(dropped, n_cv=5)
            _, _, test_concs, test_inhibs = cds.get_split(data_curves, cv)
            train_concs, train_inhibs, _, _ = cds.get_split_multiplek(data_curves, list(ks[:, cv]))
        train_data = {"lines":data_curves["lines"],
                      "drugs":data_curves["drugs"],
                      "inhibs":train_inhibs,
                      "concs":train_concs}
        test_data = {"lines":data_curves["lines"],
                      "drugs":data_curves["drugs"],
                      "inhibs":test_inhibs,
                      "concs":test_concs}
        run_2P_baselines_smoothing(train_data, test_data, cv, curves, setting=setting+suffix)
        y_pred = pd.read_csv(f"R/2P_{setting+suffix}_fold_{cv}_dropped_{dropped}.csv")["pred"].to_numpy()
        MSE = (((1 - y_pred) - test_inhibs.squeeze())**2).mean()
        perfs += [MSE]
    pd.DataFrame(perfs).to_csv(f"results/2P_{setting+suffix}_dropped_{dropped}.csv")
        
