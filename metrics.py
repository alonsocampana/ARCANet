import torchmetrics
from scipy.stats import spearmanr
from torch.nn import functional as F
metrics_detailed = torchmetrics.MetricCollection([torchmetrics.PearsonCorrCoef(),
                               torchmetrics.MeanSquaredError(),               
                               torchmetrics.MeanAbsoluteError(),
                               torchmetrics.SpearmanCorrCoef()])
metrics_rough = torchmetrics.MetricCollection([torchmetrics.MeanSquaredError(),               
                               torchmetrics.MeanAbsoluteError()])

def kl_reweighted(y_pred, y_obs, kld, eps = 0.01):
    kl_loss = kld(F.log_softmax(y_pred_9.squeeze(), 1), F.softmax(y_obs_9.squeeze(), 1))
    reweight = kl_loss.new_ones(kl_loss.shape)
    reweight[:, reweight.shape[1]-1] = 5
    return (kl_loss*reweight).sum(axis=1).mean()

def kl_loss(y_pred, y_obs, kl, eps = 0.01, mu=0.0, T = 5):
    y_pred_norm = F.log_softmax((y_pred.squeeze() - mu)*T, 1)
    y_obs_norm = F.softmax((y_obs.squeeze() - mu)*T, 1)
    return kl(y_pred_norm, y_obs_norm)

def print_average_variances(val_ds):
    unique_drugs = val_ds.data["DRUG_ID"].unique()
    rs = []
    f1s = [val_ds.data.groupby(["DRUG_ID", "COSMIC_ID"] ).mean().mean(axis=1).reset_index().query("DRUG_ID == @i") for i in unique_drugs[:30]]
    for i in range(len(unique_drugs[:30])):
        f1 = f1s[i]
        for j in range(len(unique_drugs[:30])):
            f2 = f1s[j]
            mrg = f2.merge(f1, on="COSMIC_ID")
            rs += [np.corrcoef(mrg["0_x"], mrg["0_y"])[0, 1]]

    print(np.nanmean(rs))

    unique_cosmics = val_ds.data["COSMIC_ID"].unique()

    rs = []
    f1s = [val_ds.data.groupby(["DRUG_ID", "COSMIC_ID"] ).mean().mean(axis=1).reset_index().query("COSMIC_ID == @i") for i in unique_cosmics[:30]]
    for i in range(len(unique_cosmics[:30])):
        f1 = f1s[i]
        for j in range(len(unique_cosmics[:30])):
            f2 = f1s[j]
            mrg = f2.merge(f1, on="DRUG_ID")
            rs += [np.corrcoef(mrg["0_x"], mrg["0_y"])[0, 1]]

    print(np.nanmean(rs))

def drugwise_r_variances():
    rs = []
    for drug in val_df["DRUG_ID"].unique():
        rs += [spearmanr(val_df.query("DRUG_ID == @drug")["obs_var"].to_numpy(),
                         val_df.query("DRUG_ID == @drug")["pred_var"].to_numpy())[0]]
    return np.mean(rs)

def cellwise_r_variances():
    rs = []
    for drug in val_df["COSMIC_ID"].unique():
        rs += [spearmanr(val_df.query("COSMIC_ID == @drug")["obs_var"].to_numpy(),
                         val_df.query("COSMIC_ID == @drug")["pred_var"].to_numpy())[0]]
    return np.mean(rs)