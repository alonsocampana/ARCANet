import json
import uuid
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from patsy import dmatrix
from sklearn.linear_model import Ridge
import rpy2.robjects as robjects
import warnings

class JSONLogger():
    def __init__(self, root ="logs/", path = None, save_at_iter = False):
        log = {}
        if path is None:
            path = root + str(uuid.uuid4()) + ".json"
        else:
            path = root + path
        if os.path.exists(path):
            with open(path, "r") as f:
                log = json.load(f)
        else:
            with open(path, "w") as f:
                json.dump(log, f)
        self.path = path
        self.save_at_iter = save_at_iter
        self.log = log
    def __call__(self, epoch, **kwargs):
        key_args = self.nums_as_strings(kwargs)
        epoch_log = {}
        self.log[epoch] = key_args
        if self.save_at_iter:
            self.save_log()
    def save_log(self):
        with open(self.path, "w") as f:
                json.dump(self.log, f)
    def nums_as_strings(self, args):
        key_args = {}
        for arg in list(args.keys()):
            key_args[arg] = str(args[arg])
        return key_args
    

def trapezoidal_auc(sens):
    non_na_sens = sens[~sens.isna()]
    sens_ = non_na_sens.to_numpy()
    concs_ = non_na_sens.index.to_numpy().astype(float)
    cum = 0
    max_cum = 0
    for i in range(6):
        cum += (sens_[i] + sens_[i+1])/2 * (concs_[i] + concs_[i+1])/2
        max_cum +=  1/2*(concs_[i] + concs_[i+1])
    return cum/max_cum

def get_binary_df(df, col = "z"):
    return df.assign(sensitive = (df[col] < -2).astype(int),
                resistant = (df[col] > 2).astype(int))
def get_rs(df, col):
    drugs = {d:i for i, d in enumerate(df["DRUG_ID"].unique())}
    lines = {l:i for i, l in enumerate(df["COSMIC_ID"].unique())}
    df_cats =  df.assign(L = df.loc[:, "COSMIC_ID"].map(lines).astype(int),
                                        D = df.loc[:, "DRUG_ID"].map(drugs).astype(int))
    X = dmatrix("C(D) + C(L)", df_cats)
    clf_o = Ridge(0.000001)
    clf_o.fit(X, df[col])
    y_pred_o = clf_o.predict(X)
    rs_o = df[col] - y_pred_o
    return df.assign(r = rs_o)
def get_drugwise_zs(df, col):
    mus = df.groupby("DRUG_ID")[col].mean()
    sigs = df.groupby("DRUG_ID")[col].std()
    return df.assign(z = (df.loc[:, col] - mus.loc[df.loc[:, "DRUG_ID"]].to_numpy())/sigs.loc[df.loc[:, "DRUG_ID"]].to_numpy())
class GDSC1RawDatasetInter(Dataset):
    def __init__(self,
                drugs,
                lines,
                inhibs,
                concs):
        self.drugs = drugs
        self.lines = lines
        self.inhibs = inhibs
        self.concs = concs
        drugs_screened = pd.read_csv("data/screened_compounds_rel_8.4.csv")
        self.dict_drugs = {drugs_screened["DRUG_ID"].to_numpy()[i]:drugs_screened["DRUG_NAME"].to_numpy()[i] for i in range(len(drugs_screened))}
        self.inv_dict = {it[1]:it[0] for it in self.dict_drugs.items()}
        self.graphs = torch.load("data/graphs_GDSC.pt")
        self.expression = pd.read_csv("data/lines_processed.zip", index_col=0).sort_index()
        lines = np.intersect1d(self.expression.index.to_numpy(), np.unique(self.lines))
        drugs = []
        unique_drugs = np.unique(self.drugs)
        drugs_id = unique_drugs[np.isin(pd.Series(unique_drugs.astype(int)).map(self.dict_drugs), list(self.graphs.keys()))]
        drugs_indata = np.isin(self.drugs, drugs_id)
        lines_indata = np.isin(self.lines, lines)
        self.inhibs_filtered  = self.inhibs[lines_indata&drugs_indata]
        self.concs_filtered  = self.concs[lines_indata&drugs_indata]
        self.drugs_filtered = self.drugs[lines_indata&drugs_indata]
        self.lines_filtered  = self.lines[lines_indata&drugs_indata]
        self.lid = lines_indata
        self.did = drugs_indata
    def __len__(self):
        return len(self.drugs)
    def __getitem__(self, idx):
        drug = self.graphs[self.dict_drugs[self.drugs[idx]]].clone()
        drug["expression"] = torch.Tensor(self.expression.loc[self.lines[idx]].to_numpy()[None, :])
        drug["concs"] = torch.Tensor(self.concs[idx][None, :])
        drug["inhibs"] = torch.Tensor(self.inhibs[idx][None, :])
        return drug

def R_format_df(data, max_cs, n_points=None):
    if n_points is None:
        n_points = data["concs"].shape[1]
    return pd.DataFrame({"drug":np.repeat(data["drugs"], n_points),
                    "CL":np.repeat(data["lines"], n_points),
                    "x":data["concs"].flatten(),
                    "y":1 - np.clip(data["inhibs"].flatten(), 0, 1),
                    "maxc":np.repeat(max_cs.loc[data["drugs"]].to_numpy(), n_points)})
    
def run_R_baselines(train_data, test_data, cv, setting = "extrapolation"):
    max_concs = curves.iloc[:, 2:].columns.to_numpy()[(~curves.iloc[:, 2:].isna()).to_numpy().argmax(axis=1)]
    max_cs = curves.loc[:,["DRUG_ID"]].assign(max_concs = max_concs).groupby("DRUG_ID").max()
    test_df = R_format_df(test_data, max_cs)
    train_df = R_format_df(train_data, max_cs)
    R_train = pd.concat([train_df, test_df.query("x != 1")])
    R_train["x"] = R_train["x"].map({7:1, 6:2, 5:3, 4:4, 3:5, 2:6, 1:7})
    R_test = test_df.query("x == 1")
    R_test["x"] = R_test["x"].map({7:1, 6:2, 5:3, 4:4, 3:5, 2:6, 1:7})
    R_train.to_csv("R/temp_train.csv", index=False)
    R_test.to_csv("R/temp_test.csv", index=False)
    r_source = robjects.r['source']
    try:
        r_source('R/sigmoid2P.R')
        y_pred = pd.read_csv("R/preds.csv")
        y_pred.to_csv(f"R/2P_{setting}_fold_{cv}.csv")
    except Exception as E:
        print(E)
        warnings.warn("The fitting of model 2P failed")
    try:
        r_source('R/sigmoid3P.R')
        y_pred = pd.read_csv("R/preds.csv")
        y_pred.to_csv(f"R/3P_{setting}_fold_{cv}.csv")
    except Exception as E:
        print(E)
        warnings.warn("The fitting of model 3P failed")
    try:
        r_source('R/sigmoid4P.R')
        y_pred = pd.read_csv("R/preds.csv")
        y_pred.to_csv(f"R/4P_{setting}_fold_{cv}.csv")
    except Exception as E:
        print(E)
        warnings.warn("The fitting of model 4P failed")

def run_2P_baselines_extrapolation(train_data, test_data, cv, setting = "extrapolation"):
    max_concs = curves.iloc[:, 2:].columns.to_numpy()[(~curves.iloc[:, 2:].isna()).to_numpy().argmax(axis=1)]
    max_cs = curves.loc[:,["DRUG_ID"]].assign(max_concs = max_concs).groupby("DRUG_ID").max()
    test_df = R_format_df(test_data, max_cs)
    train_df = R_format_df(train_data, max_cs)
    R_train = pd.concat([train_df, test_df.query("x != 1")])
    R_train["x"] = R_train["x"].map({7:1, 6:2, 5:3, 4:4, 3:5, 2:6, 1:7})
    R_test = test_df.query("x == 1")
    R_test["x"] = R_test["x"].map({7:1, 6:2, 5:3, 4:4, 3:5, 2:6, 1:7})
    R_train.to_csv("R/temp_train.csv", index=False)
    R_test.to_csv("R/temp_test.csv", index=False)
    r_source = robjects.r['source']
    try:
        r_source('R/sigmoid2P.R')
        y_pred = pd.read_csv("R/preds.csv")
        y_pred.to_csv(f"R/2P_{setting}_fold_{cv}.csv")
    except Exception as E:
        print(E)
        warnings.warn("The fitting of model 2P failed")

def run_2P_baselines_smoothing(train_data, test_data, cv, curves, setting = "smoothing", dropped=1):
    max_concs = curves.iloc[:, 2:].columns.to_numpy()[(~curves.iloc[:, 2:].isna()).to_numpy().argmax(axis=1)]
    max_cs = curves.loc[:,["DRUG_ID"]].assign(max_concs = max_concs).groupby("DRUG_ID").max()
    test_df = R_format_df(test_data, max_cs)
    train_df = R_format_df(train_data, max_cs)
    R_train = train_df
    R_train["x"] = R_train["x"].map({7:1, 6:2, 5:3, 4:4, 3:5, 2:6, 1:7})
    R_test = test_df
    R_test["x"] = R_test["x"].map({7:1, 6:2, 5:3, 4:4, 3:5, 2:6, 1:7})
    R_train.to_csv("R/temp_train.csv", index=False)
    R_test.to_csv("R/temp_test.csv", index=False)
    r_source = robjects.r['source']
    try:
        r_source('R/sigmoid2P.R')
        y_pred = pd.read_csv("R/preds.csv")
        y_pred.to_csv(f"R/2P_{setting}_fold_{cv}_dropped_{dropped}.csv")
    except Exception as E:
        print(E)
        warnings.warn("The fitting of model 2P failed")
        
class CurvesPreprocessor():
    def __init__(self,
                 reindex = True,
                 filter_missing = True,
                 drop_data_ids_10x = None,
                 drop_at_random_10x = None,
                 drop_data_drugs_10x = None,
                 n_seed = 3558,):
        self.reindex = reindex
        self.drop_data_ids = drop_data_ids_10x
        self.drop_at_random = drop_at_random_10x
        self.drop_drugs = drop_data_drugs_10x
        self.filter_missing = filter_missing
        self.n_seed = n_seed
    def __call__(self, x):
        curves = x.copy()
        if self.drop_at_random is not None:
            np.random.seed(self.n_seed)
            ids = np.array_split(np.random.permutation(curves.index.to_numpy()), 10)
            to_drop = np.concatenate(ids[:self.drop_at_random])
            curves = curves.drop(to_drop, axis=0)
        if self.drop_drugs is not None:
            np.random.seed(self.n_seed)
            drug_splits = np.array_split(np.random.permutation(curves["DRUG_ID"].unique()), 10)
            drugs_to_drop = np.concatenate(drug_splits[:self.drop_drugs])
            curves = curves.query("DRUG_ID not in @drugs_to_drop")
        if self.drop_data_ids is not None:
            np.random.seed(self.n_seed)
            drug_splits = np.array_split(np.random.permutation(curves["DRUG_ID"].unique()), 10)
            line_splits = np.array_split(np.random.permutation(curves["COSMIC_ID"].unique()), 10)
            drugs_to_drop = np.concatenate(drug_splits[:self.drop_data_ids])
            lines_to_drop = np.concatenate(line_splits[:self.drop_data_ids])
            curves = curves.query("DRUG_ID not in @drugs_to_drop & COSMIC_ID not in @lines_to_drop")
        unique_drugs = curves["DRUG_ID"].unique()
        self.unique_drugs = unique_drugs
        drugs_dict = {unique_drugs[i]:i for i in range(len(unique_drugs))}
        self.drugs_dict = drugs_dict
        unique_lines = curves["COSMIC_ID"].unique()
        self.unique_lines = unique_lines
        lines_dict = {unique_lines[i]:i for i in range(len(unique_lines))}
        self.lines_dict = lines_dict
        if self.filter_missing:
            drugs_screened = pd.read_csv("data/screened_compounds_rel_8.4.csv")
            dict_drugs = {drugs_screened["DRUG_ID"].to_numpy()[i]:drugs_screened["DRUG_NAME"].to_numpy()[i] for i in range(len(drugs_screened))}
            inv_dict = {it[1]:it[0] for it in dict_drugs.items()}
            lines = pd.read_csv("data/lines_processed.zip", index_col=0).sort_index()
            graphs = torch.load("data/graphs_GDSC.pt")
            lines_indata = np.isin(unique_lines.astype(int), lines.index.to_numpy())
            keep_lines = unique_lines[lines_indata] 
            drugs_indata = np.isin(pd.Series(unique_drugs.astype(int)).map(dict_drugs), list(graphs.keys()))
            keep_drugs = unique_drugs[drugs_indata]
            curves = curves.query("(DRUG_ID in @keep_drugs)&(COSMIC_ID in @keep_lines)")   
        if self.reindex:
            curves["COSMIC_ID"] = curves["COSMIC_ID"].map(lines_dict)
            curves["DRUG_ID"] = curves["DRUG_ID"].map(drugs_dict)
        curves = curves[curves.isna().sum(axis=1) == 224]
        return curves

class CurveDataSplitter():
    def __init__(self,
                 curves,
                 clip = False,
                 concs_as_dilution_step = True,
                 smoothing = True,
                 drop_random_training = None,
                 drop_classes_training = None,
                 n_seed = 3558):
        self.curves = curves
        self.L, self.C = curves.shape
        self.cv = None
        self.concs_as_dilution_step = concs_as_dilution_step
        self.clip = clip
        n_drugs = int(curves["DRUG_ID"].max() + 1)
        self.drop_random_training = drop_random_training
        self.drop_classes_training = drop_classes_training
        self.n_seed = n_seed
        if smoothing:
            self.generate_cv()
        else:
            self.generate_cv_interpolation(n_drugs)
    def __call__(self, x):
        pass
    def generate_cv(self):
        np.random.seed(self.n_seed)
        self.cv = np.array([np.random.permutation(np.arange(7)) for x in range(self.L)])
    def generate_cv_interpolation(self, n_drugs):
        np.random.seed(self.n_seed)
        drug_perms = {d:np.random.permutation(np.arange(1, 6))for d in range(n_drugs)}
        data_curves = self.get_data(self.curves)
        self.cv = np.vstack(pd.Series(data_curves["drugs"]).map(drug_perms).tolist())
    def get_data(self, x):
        curves = x.copy()
        if self.drop_random_training is not None:
            np.random.seed(self.n_seed)
            ids = np.array_split(np.random.permutation(np.arange(len(curves))), 10)
            to_drop = np.concatenate(ids[:self.drop_random_training])
            mask = np.ones(len(curves))
            mask[to_drop] = 0
            self.random_mask = mask.astype(bool)
        cols = curves.iloc[:, 2:].columns.to_numpy().astype(float)
        drugs = curves["DRUG_ID"].to_numpy()
        lines = curves["COSMIC_ID"].to_numpy()
        curv = curves.iloc[:, 2:].to_numpy()
        indices = np.tile(np.arange(curv.shape[1])[None, :], [len(curves), 1])[~np.isnan(curv)]
        concs = curves.iloc[:, 2:].columns.to_numpy().astype(float)[indices].reshape(len(curves), -1)
        if self.concs_as_dilution_step:
            concs = np.tile(np.arange(1, 8)[::-1].astype(float), [concs.shape[0], 1])
        inhibs = curves.iloc[:, 2:].to_numpy()[~np.isnan(curv)].reshape(len(curves), -1)
        if self.clip:
            inhibs = np.clip(inhibs, 0, 1)
        return {"drugs": drugs,
               "lines": lines,
               "concs": concs,
                "inhibs": inhibs}
    def get_doublek(self):
        np.random.seed(self.n_seed)
        correct = False
        while not correct:
            second_drop = np.random.permutation(np.arange(7))
            first_drop = np.arange(7)
            correct = ~(first_drop == second_drop).any()
        return np.vstack([first_drop, second_drop])
    def get_triplek(self):
        np.random.seed(self.n_seed)
        correct = False
        while not correct:
            third_drop = np.random.permutation(np.arange(7))
            second_drop = np.random.permutation(np.arange(7))
            first_drop = np.arange(7)
            correct = ~((first_drop == second_drop)|(first_drop == third_drop)|(third_drop == second_drop)).any()
        return np.vstack([first_drop, second_drop, third_drop])
    def get_multiplek(self, n_k, n_cv = 5):
        np.random.seed(self.n_seed)
        correct = False
        while not correct:
            correct = True
            drops = [np.random.permutation(np.arange(n_cv)) for i in range(n_k-1)]
            first_drop = [np.arange(n_cv)]
            all_drops = drops+first_drop
            for i in np.vstack(all_drops).T:
                if len(set(i)) < n_k:
                    correct = False
                    break
        return np.vstack(all_drops)
    def get_split(self, x, k):
        curves = self.curves
        lo = self.cv[:, k]
        concs = x["concs"].copy()
        inhibs = x["inhibs"].copy()
        test_concs = concs[np.arange(len(concs)), lo]
        test_inhibs = inhibs[np.arange(len(concs)), lo]
        concs[np.arange(len(concs)), lo] = np.nan
        inhibs[np.arange(len(concs)), lo] = np.nan
        train_concs = concs.flatten()[~np.isnan(concs.flatten())].reshape(len(concs), -1)
        train_inhibs = inhibs.flatten()[~np.isnan(concs.flatten())].reshape(len(curves), -1)
        return train_concs, train_inhibs, test_concs[:, None], test_inhibs[:, None]
    def get_split_multiplek(self, x, ks):
        test_concs = []
        test_inhibs = []
        concs = x["concs"].copy()
        inhibs = x["inhibs"].copy()
        for k in ks:
            lo = self.cv[:, k]
            test_concs += [concs[np.arange(len(concs)), lo][:, None]]
            test_inhibs += [inhibs[np.arange(len(concs)), lo][:, None]]
            concs[np.arange(len(concs)), lo] = np.nan
            inhibs[np.arange(len(concs)), lo] = np.nan
        train_concs = concs.flatten()[~np.isnan(concs.flatten())].reshape(len(concs), -1)
        train_inhibs = inhibs.flatten()[~np.isnan(concs.flatten())].reshape(len(concs), -1)
        return train_concs, train_inhibs, np.concatenate(test_concs, axis=1), np.concatenate(test_inhibs, axis=1)
    def get_split_dd(self, x, k, k_folds = 10):
        test_concs = []
        test_inhibs = []
        concs = x["concs"].copy()
        inhibs = x["inhibs"].copy()
        drugs = x["drugs"].copy()
        lines = x["lines"].copy()
        np.random.seed(self.n_seed)
        permuted_drugs = np.random.permutation(np.unique(drugs))
        drugs_test = np.array_split(permuted_drugs, k_folds)[k]
        test_mask = np.isin(drugs, drugs_test)
        if self.drop_classes_training is not None:
            np.random.seed(self.n_seed)
            class_folds = np.arange(k_folds)
            class_folds = np.delete(class_folds, k)
            remove_from_training = list(np.random.choice(class_folds, self.drop_classes_training))
            remove_from_training += [k]
            drugs_ommited = np.concatenate([np.array_split(permuted_drugs,k_folds)[drop_k] for drop_k in remove_from_training])
            ommited_mask = np.isin(drugs, drugs_ommited)
        else:
            ommited_mask = test_mask
        train_data = {"drugs":drugs[~ommited_mask],
                      "lines":lines[~ommited_mask],
                      "concs":concs[~ommited_mask],
                      "inhibs":inhibs[~ommited_mask]}
        test_data = {"drugs":drugs[test_mask],
                      "lines":lines[test_mask],
                      "concs":concs[test_mask],
                      "inhibs":inhibs[test_mask]}
        return train_data, test_data
    def get_split_po(self, x, k, k_folds = 10):
        test_concs = []
        test_inhibs = []
        concs = x["concs"].copy()
        inhibs = x["inhibs"].copy()
        drugs = x["drugs"].copy()
        lines = x["lines"].copy()
        np.random.seed(self.n_seed)
        lines_test = np.array_split(np.random.permutation(np.unique(lines)), k_folds)[k]
        test_mask = np.isin(lines, lines_test)
        train_data = {"drugs":drugs[~test_mask],
                      "lines":lines[~test_mask],
                      "concs":concs[~test_mask],
                      "inhibs":inhibs[~test_mask]}
        test_data = {"drugs":drugs[test_mask],
                      "lines":lines[test_mask],
                      "concs":concs[test_mask],
                      "inhibs":inhibs[test_mask]}
        return train_data, test_data
    
class GDSC1RawDatasetInter(Dataset):
    def __init__(self,
                drugs,
                lines,
                inhibs,
                concs):
        self.drugs = drugs
        self.lines = lines
        self.inhibs = inhibs
        self.concs = concs
        drugs_screened = pd.read_csv("data/screened_compounds_rel_8.4.csv")
        self.dict_drugs = {drugs_screened["DRUG_ID"].to_numpy()[i]:drugs_screened["DRUG_NAME"].to_numpy()[i] for i in range(len(drugs_screened))}
        self.inv_dict = {it[1]:it[0] for it in self.dict_drugs.items()}
        self.graphs = torch.load("data/graphs_GDSC.pt")
        self.expression = pd.read_csv("data/lines_processed.zip", index_col=0).sort_index()
        lines = np.intersect1d(self.expression.index.to_numpy(), np.unique(self.lines))
        drugs = []
        unique_drugs = np.unique(self.drugs)
        drugs_id = unique_drugs[np.isin(pd.Series(unique_drugs.astype(int)).map(self.dict_drugs), list(self.graphs.keys()))]
        drugs_indata = np.isin(self.drugs, drugs_id)
        lines_indata = np.isin(self.lines, lines)
        self.inhibs_filtered  = self.inhibs[lines_indata&drugs_indata]
        self.concs_filtered  = self.concs[lines_indata&drugs_indata]
        self.drugs_filtered = self.drugs[lines_indata&drugs_indata]
        self.lines_filtered  = self.lines[lines_indata&drugs_indata]
        self.lid = lines_indata
        self.did = drugs_indata
    def __len__(self):
        return len(self.drugs)
    def __getitem__(self, idx):
        drug = self.graphs[self.dict_drugs[self.drugs[idx]]].clone()
        drug["expression"] = torch.Tensor(self.expression.loc[self.lines[idx]].to_numpy()[None, :])
        drug["concs"] = torch.Tensor(self.concs[idx][None, :])
        drug["inhibs"] = torch.Tensor(self.inhibs[idx][None, :])
        return drug
    
class GDSC1RawDatasetExtrap(Dataset):
    def __init__(self,
                drugs,
                lines,
                inhibs,
                concs,
                test_drugs,
                test_lines,
                test_inhibs,
                test_concs):
        test_inhibs = test_inhibs.copy()
        test_inhibs[:, 6] = np.NaN
        self.drugs = np.concatenate([drugs, test_drugs])
        self.lines = np.concatenate([lines, test_lines])
        self.inhibs = np.concatenate([inhibs, test_inhibs])
        self.concs = np.concatenate([concs, test_concs])
        drugs_screened = pd.read_csv("data/screened_compounds_rel_8.4.csv")
        self.dict_drugs = {drugs_screened["DRUG_ID"].to_numpy()[i]:drugs_screened["DRUG_NAME"].to_numpy()[i] for i in range(len(drugs_screened))}
        self.inv_dict = {it[1]:it[0] for it in self.dict_drugs.items()}
        self.graphs = torch.load("data/graphs_GDSC.pt")
        self.expression = pd.read_csv("data/lines_processed.zip", index_col=0).sort_index()
        lines = np.intersect1d(self.expression.index.to_numpy(), np.unique(self.lines))
        drugs = []
        unique_drugs = np.unique(self.drugs)
        drugs_id = unique_drugs[np.isin(pd.Series(unique_drugs.astype(int)).map(self.dict_drugs), list(self.graphs.keys()))]
        drugs_indata = np.isin(self.drugs, drugs_id)
        lines_indata = np.isin(self.lines, lines)
        self.inhibs_filtered  = self.inhibs[lines_indata&drugs_indata]
        self.concs_filtered  = self.concs[lines_indata&drugs_indata]
        self.drugs_filtered = self.drugs[lines_indata&drugs_indata]
        self.lines_filtered  = self.lines[lines_indata&drugs_indata]
        self.lid = lines_indata
        self.did = drugs_indata
    def __len__(self):
        return len(self.drugs)
    def __getitem__(self, idx):
        drug = self.graphs[self.dict_drugs[self.drugs[idx]]].clone()
        drug["expression"] = torch.Tensor(self.expression.loc[self.lines[idx]].to_numpy()[None, :])
        if np.isnan(self.inhibs[idx]).any():
            select_idx = np.arange(0, 6)
        else:
            select_idx = np.arange(0, 7)
            drop_idx = np.random.choice(select_idx)
            select_idx = np.delete(select_idx, drop_idx)
        drug["concs"] = torch.Tensor(self.concs[idx][select_idx][None, :])
        drug["inhibs"] = torch.Tensor(self.inhibs[idx][select_idx][None, :])
        return drug

class GDSC1RawDatasetEpoint(Dataset):
    def __init__(self,
                drugs,
                lines,
                inhibs,
                concs):
        self.drugs = drugs
        self.lines = lines
        self.inhibs = inhibs[:, 6]
        self.concs = concs[:, 6]
        drugs_screened = pd.read_csv("data/screened_compounds_rel_8.4.csv")
        self.dict_drugs = {drugs_screened["DRUG_ID"].to_numpy()[i]:drugs_screened["DRUG_NAME"].to_numpy()[i] for i in range(len(drugs_screened))}
        self.inv_dict = {it[1]:it[0] for it in self.dict_drugs.items()}
        self.graphs = torch.load("data/graphs_GDSC.pt")
        self.expression = pd.read_csv("data/lines_processed.zip", index_col=0).sort_index()
        lines = np.intersect1d(self.expression.index.to_numpy(), np.unique(self.lines))
        drugs = []
        unique_drugs = np.unique(self.drugs)
        drugs_id = unique_drugs[np.isin(pd.Series(unique_drugs.astype(int)).map(self.dict_drugs), list(self.graphs.keys()))]
        drugs_indata = np.isin(self.drugs, drugs_id)
        lines_indata = np.isin(self.lines, lines)
        self.inhibs_filtered  = self.inhibs[lines_indata&drugs_indata]
        self.concs_filtered  = self.concs[lines_indata&drugs_indata]
        self.drugs_filtered = self.drugs[lines_indata&drugs_indata]
        self.lines_filtered  = self.lines[lines_indata&drugs_indata]
        self.lid = lines_indata
        self.did = drugs_indata
    def __len__(self):
        return len(self.drugs)
    def __getitem__(self, idx):
        drug = self.graphs[self.dict_drugs[self.drugs[idx]]].clone()
        drug["expression"] = torch.Tensor(self.expression.loc[self.lines[idx]].to_numpy()[None, :])
        drug["concs"] = torch.Tensor(self.concs[[idx]][None, :])
        drug["inhibs"] = torch.Tensor(self.inhibs[[idx]][None, :])
        return drug