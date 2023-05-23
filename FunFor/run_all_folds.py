import torch
import pandas as pd
import numpy as np
import sys
from utils import CurvesPreprocessor, CurvesPreprocessor
import rpy2.robjects as robjects

drug_features = pd.DataFrame(torch.load("GDSC2_morganFingerprint_R2_{}.pt")).T
drug_features2 = pd.DataFrame(torch.load("GDSC1_morganFingerprint_R2_{}.pt")).T
drug_features = pd.concat([drug_features,drug_features2]).drop_duplicates()
ids = pd.read_csv("gdscidtoname.csv", index_col=0)
merged = drug_features.reset_index().merge(ids, left_on="index", right_on = "DRUG_NAME")
table_drugs = merged.drop(["index", "DRUG_NAME"], axis=1).set_index("DRUG_ID").drop_duplicates()
drugs_data = table_drugs.index.unique()
table_lines = pd.read_csv("lines_processed.csv", index_col=0)
lines_data = table_lines.index.unique()
for cv in range(1, 10):
    inhibs_train = pd.read_csv(f"data_baseline/po_train_{cv}.csv", index_col=0).astype({"DRUG_ID":int}).query("DRUG_ID in @drugs_data & COSMIC_ID in @lines_data")
    inhibs_test = pd.read_csv(f"data_baseline/po_test_{cv}.csv", index_col=0).astype({"DRUG_ID":int}).query("DRUG_ID in @drugs_data & COSMIC_ID in @lines_data")
    drug_f = table_drugs.loc[inhibs_train["DRUG_ID"].astype(int)]
    cell_f = table_lines.loc[inhibs_train["COSMIC_ID"].astype(int)]
    X_train = pd.concat([drug_f.reset_index(drop=True), cell_f.reset_index(drop=True)], axis=1, ignore_index=True).to_numpy()
    drug_f = table_drugs.loc[inhibs_test["DRUG_ID"].astype(int)]
    cell_f = table_lines.loc[inhibs_test["COSMIC_ID"].astype(int)]
    X_test = pd.concat([drug_f.reset_index(drop=True), cell_f.reset_index(drop=True)], axis=1, ignore_index=True).to_numpy()
    keep_vars = X_train.var(axis=0) > 0.015
    X_train = X_train[:, keep_vars]
    X_test = X_test[:, keep_vars]
    Y_train = inhibs_train.iloc[:, 2:]
    Y_test = inhibs_test.iloc[:, 2:]
    pd.DataFrame(X_train).to_csv("X_train.csv", index=False)
    pd.DataFrame(X_test).to_csv("X_test.csv", index=False)
    pd.DataFrame(Y_train).to_csv("y_train.csv", index=False)
    pd.DataFrame(Y_test).to_csv("y_test.csv", index=False)
    r_source = robjects.r['source']
    r_source('fit_data.R')
    y_pred = pd.read_csv("y_pred.csv", index_col=0)
    y_pred.to_csv(f"y_pred_po_{cv}.csv")

for cv in range(10):
    inhibs_train = pd.read_csv(f"data_baseline/dd_train_{cv}.csv", index_col=0).astype({"DRUG_ID":int}).query("DRUG_ID in @drugs_data & COSMIC_ID in @lines_data")
    inhibs_test = pd.read_csv(f"data_baseline/dd_test_{cv}.csv", index_col=0).astype({"DRUG_ID":int}).query("DRUG_ID in @drugs_data & COSMIC_ID in @lines_data")
    drug_f = table_drugs.loc[inhibs_train["DRUG_ID"].astype(int)]
    cell_f = table_lines.loc[inhibs_train["COSMIC_ID"].astype(int)]
    X_train = pd.concat([drug_f.reset_index(drop=True), cell_f.reset_index(drop=True)], axis=1, ignore_index=True).to_numpy()
    drug_f = table_drugs.loc[inhibs_test["DRUG_ID"].astype(int)]
    cell_f = table_lines.loc[inhibs_test["COSMIC_ID"].astype(int)]
    X_test = pd.concat([drug_f.reset_index(drop=True), cell_f.reset_index(drop=True)], axis=1, ignore_index=True).to_numpy()
    keep_vars = X_train.var(axis=0) > 0.015
    X_train = X_train[:, keep_vars]
    X_test = X_test[:, keep_vars]
    Y_train = inhibs_train.iloc[:, 2:]
    Y_test = inhibs_test.iloc[:, 2:]
    pd.DataFrame(X_train).to_csv("X_train.csv", index=False)
    pd.DataFrame(X_test).to_csv("X_test.csv", index=False)
    pd.DataFrame(Y_train).to_csv("y_train.csv", index=False)
    pd.DataFrame(Y_test).to_csv("y_test.csv", index=False)
    r_source = robjects.r['source']
    r_source('fit_data.R')
    y_pred = pd.read_csv("y_pred.csv", index_col=0)
    y_pred.to_csv(f"y_pred_dd_{cv}.csv")