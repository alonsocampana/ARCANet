import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from statannotations.Annotator import Annotator
from scipy.stats import ttest_rel

def get_last_performance_json(json, n_iter = 100):
    return eval(pd.read_json(json).T.iloc[n_iter]["test"])["MeanSquaredError"]
#Create plots learning curves
training = []
testing = []
for cv in range(7):
    log = pd.read_json(f"logs/GDSC2/smoothing_fold_{cv}_dropped_1_loss_klmax_iter_300.json").T.iloc[1:]
    training+= [log["train"].apply(lambda x: eval(x)["MeanSquaredError"]).to_numpy()]
    testing+= [log["test"].apply(lambda x: eval(x)["MeanSquaredError"]).to_numpy()]
training_curve = pd.DataFrame(training)
testing_curve = pd.DataFrame(testing)
training_curve = pd.DataFrame(training).stack().reset_index()
training_curve.columns = ["fold", "epoch", "MSE"]
testing_curve = pd.DataFrame(testing).stack().reset_index()
testing_curve.columns = ["fold", "epoch", "MSE"]
plt.figure()
p = sns.lineplot(x = "epoch", y = "MSE", data = testing_curve, label="test set")
p = sns.lineplot(x = "epoch", y = "MSE", data = training_curve, label="train set")
p.set_ylim(0.001, 0.01)
plt.vlines(100, 0, 0.2, color = "red", linestyle = "--", linewidth = 1, label="utilized configuration")
p.legend()
plt.savefig("plots/training_curves.pdf", dpi=300, bbox_inches = "tight")
plt.show()
plt.close()
plt.cla()
plt.clf()
#create plots missing data
palette = sns.color_palette(n_colors = 6)
perfs_00 = [get_last_performance_json(f"logs/GDSC2/smoothing_fold_{cv}_dropped_1_loss_klmax_iter_300.json") for cv in range(7)]
perfs_10 = [get_last_performance_json(f"logs/GDSC2/smoothing_fold_{cv}_dropped_1_loss_kl_random_1.json") for cv in range(7)]
perfs_20 = [get_last_performance_json(f"logs/GDSC2/smoothing_fold_{cv}_dropped_1_loss_kl_random_2.json") for cv in range(7)]
perfs_40 = [get_last_performance_json(f"logs/GDSC2/smoothing_fold_{cv}_dropped_1_loss_kl_random_4.json") for cv in range(7)]
perfs_60 = [get_last_performance_json(f"logs/GDSC2/smoothing_fold_{cv}_dropped_1_loss_kl_random_6.json") for cv in range(7)]
perfs_80 = [get_last_performance_json(f"logs/GDSC2/smoothing_fold_{cv}_dropped_1_loss_kl_random_8.json") for cv in range(7)]
p0_s = pd.read_csv("results/GDSC2p_smoothing.csv").query("dropped == 0").to_numpy()[:, 3]
p1_s = pd.read_csv("results/2P_smoothing_random_1_dropped_1.csv").to_numpy()[:, 1]
p2_s = pd.read_csv("results/2P_smoothing_random_2_dropped_1.csv").to_numpy()[:, 1]
p4_s = pd.read_csv("results/2P_smoothing_random_4_dropped_1.csv").to_numpy()[:, 1]
p6_s = pd.read_csv("results/2P_smoothing_random_6_dropped_1.csv").to_numpy()[:, 1]
p8_s = pd.read_csv("results/2P_smoothing_random_8_dropped_1.csv").to_numpy()[:, 1]
dfs_dl_random = pd.concat([pd.DataFrame(perfs_00).assign(training_data = 100),
    pd.DataFrame(perfs_10).assign(training_data = 90),
pd.DataFrame(perfs_20).assign(training_data = 80),
pd.DataFrame(perfs_40).assign(training_data = 60),
pd.DataFrame(perfs_60).assign(training_data = 40),
pd.DataFrame(perfs_80).assign(training_data = 20)])
dfs_sig_random = pd.concat([pd.DataFrame(p0_s).assign(training_data = 100),
    pd.DataFrame(p1_s).assign(training_data = 90),
pd.DataFrame(p2_s).assign(training_data = 80),
pd.DataFrame(p4_s).assign(training_data = 60),
pd.DataFrame(p6_s).assign(training_data = 40),
pd.DataFrame(p8_s).assign(training_data = 20)])
p = sns.lineplot(x = "training_data", y=0, data = dfs_dl_random.reset_index(), ci=95, err_style='bars', color=palette[4])
p = sns.lineplot(x = "training_data", y=0, data = dfs_sig_random.reset_index(), ci=95, err_style='bars', color=palette[2])
p.set_xticks([20, 40, 60, 80, 90, 100])
p.set_ylabel("MSE")
p.set_xlabel("Fraction of data used during training(%)")
p.legend([
          "ARCANet (KL)",
          "2P mixed-effect logistic model",], loc="upper right")
plt.savefig("plots/random_missing.pdf", dpi=300, bbox_inches = "tight")
plt.show()
plt.close()
plt.cla()
plt.clf()
perfs_10 = [get_last_performance_json(f"logs/GDSC2/smoothing_fold_{cv}_dropped_1_loss_kl_classes_1.json") for cv in range(7)]
perfs_20 = [get_last_performance_json(f"logs/GDSC2/smoothing_fold_{cv}_dropped_1_loss_kl_classes_2.json") for cv in range(7)]
perfs_40 = [get_last_performance_json(f"logs/GDSC2/smoothing_fold_{cv}_dropped_1_loss_kl_classes_4.json") for cv in range(7)]
perfs_60 = [get_last_performance_json(f"logs/GDSC2/smoothing_fold_{cv}_dropped_1_loss_kl_classes_6.json") for cv in range(7)]
perfs_30 = [get_last_performance_json(f"logs/GDSC2/smoothing_fold_{cv}_dropped_1_loss_kl_classes_3.json") for cv in range(7)]
perfs_80 = [get_last_performance_json(f"logs/GDSC2/smoothing_fold_{cv}_dropped_1_loss_kl_classes_8.json") for cv in range(7)]
p1_s = pd.read_csv("results/2P_smoothing_classes_1_dropped_1.csv").to_numpy()[:, 1]
p2_s = pd.read_csv("results/2P_smoothing_classes_2_dropped_1.csv").to_numpy()[:, 1]
p4_s = pd.read_csv("results/2P_smoothing_classes_4_dropped_1.csv").to_numpy()[:, 1]
p6_s = pd.read_csv("results/2P_smoothing_classes_6_dropped_1.csv").to_numpy()[:, 1]
p3_s = pd.read_csv("results/2P_smoothing_classes_3_dropped_1.csv").to_numpy()[:, 1]
p8_s = pd.read_csv("results/2P_smoothing_classes_8_dropped_1.csv").to_numpy()[:, 1]
dfs_dl_random = pd.concat([pd.DataFrame(perfs_00).assign(training_data = 100),
    pd.DataFrame(perfs_10).assign(training_data = 90),
pd.DataFrame(perfs_20).assign(training_data = 80),
pd.DataFrame(perfs_30).assign(training_data = 70),
pd.DataFrame(perfs_40).assign(training_data = 60),
pd.DataFrame(perfs_60).assign(training_data = 40),
pd.DataFrame(perfs_80).assign(training_data = 20)])
dfs_sig_random = pd.concat([pd.DataFrame(p0_s).assign(training_data = 100),
    pd.DataFrame(p1_s).assign(training_data = 90),
pd.DataFrame(p2_s).assign(training_data = 80),
pd.DataFrame(p3_s).assign(training_data = 70),
pd.DataFrame(p4_s).assign(training_data = 60),
pd.DataFrame(p6_s).assign(training_data = 40),
pd.DataFrame(p8_s).assign(training_data = 20)])
p = sns.lineplot(x = "training_data", y=0, data = dfs_dl_random.reset_index(), ci=95, err_style='bars', color=palette[4])
p = sns.lineplot(x = "training_data", y=0, data = dfs_sig_random.reset_index(), ci=95, err_style='bars', color=palette[2])
p.set_xticks([100, 90, 80, 70, 60, 40, 20])
p.set_ylabel("MSE")
p.set_xlabel("Fraction of drugs&cell-lines used during training(%)")
p.legend(["ARCANet (KL)",
          "2P mixed-effect logistic model"], loc="upper right")
plt.savefig("plots/systematically_missing.pdf", dpi=300, bbox_inches = "tight")
plt.show()
plt.close()
plt.cla()
plt.clf()
all_perfs = []
for emb in [64, 128, 512]:
    all_perfs += [pd.DataFrame(get_last_performance_json(f"logs/GDSC2/ablation/smoothing_fold_{cv}_loss_kl_{emb}_9_6_2.5_0.25_0.05_3_True.json")for cv in range(7)).assign(embedding_size = emb)]

complete_model = pd.DataFrame(get_last_performance_json(f"logs/GDSC2/smoothing_fold_{cv}_dropped_1_loss_klmax_iter_300.json")for cv in range(7))
all_perfs+= [complete_model.assign(embedding_size = 256)]
all_perfs = pd.concat(all_perfs)
p = sns.lineplot(x = "embedding_size", y=0, data = all_perfs.reset_index(), ci=95, err_style='bars')
p.set_ylabel("MSE")
p.set_xlabel("Embedding size")
plt.vlines(256, 0, 0.0075, color = "red", linewidth = 1, linestyle = "--",)
p.set_ylim(0.0029, 0.0040)
p.set_xticks([64, 126, 256, 512])

# Create ablation plots
plt.savefig("plots/ablation_embed_dim.pdf", dpi=300, bbox_inches = "tight")
plt.show()
plt.close()
plt.cla()
plt.clf()
all_perfs = []
for g in [0.0, 0.1]:
    all_perfs += [pd.DataFrame(get_last_performance_json(f"logs/GDSC2/ablation/smoothing_fold_{cv}_loss_kl_256_9_6_2.5_0.25_{g}_3_True.json")for cv in range(7)).assign(gaussian_variance = g)]
all_perfs+= [complete_model.assign(gaussian_variance = 0.05)]
all_perfs = pd.concat(all_perfs)

p = sns.lineplot(x = "gaussian_variance", y=0, data = all_perfs.reset_index(), ci=95, err_style='bars', label = "test set")
p.set_ylabel("MSE")
p.set_xlabel("Gaussian noise for drug concentrations $(\sigma_c)$")
plt.vlines(0.05, 0, 0.0075, color = "red", linewidth = 1, linestyle = "--",label="utilized configuration")
p.set_ylim(0.0029, 0.0040)
p.set_xticks([0, 0.05, 0.1])
plt.legend()
plt.savefig("plots/ablation_gaussian_noise.pdf", dpi=300, bbox_inches = "tight")
plt.show()
plt.close()
plt.cla()
plt.clf()
all_perfs = []
for ly in [1, 2, 5, 7, 8, 9, 10, 11]:
    all_perfs += [pd.DataFrame(get_last_performance_json(f"logs/GDSC2/ablation/smoothing_fold_{cv}_loss_kl_256_{ly}_6_2.5_0.25_0.05_3_True.json")for cv in range(7)).assign(layer_number = ly)]

pd.DataFrame(get_last_performance_json(f"logs/GDSC2/ablation/smoothing_fold_{cv}_loss_kl_256_9_6_2.5_0.25_0.05_3_False.json", 100)for cv in range(7)).mean()

all_perfs += [complete_model.assign(layer_number = 9)]
all_perfs = pd.concat(all_perfs)

p = sns.lineplot(x = "layer_number", y=0, data = all_perfs.reset_index(), ci=95, err_style='bars')
p.set_ylabel("MSE")
p.set_xlabel("Number of graph attention layers")
plt.vlines(9, 0, 0.0075, color = "red", linewidth = 1, linestyle = "--",)
p.set_ylim(0.0029, 0.0040)
p.set_xticks([2, 5, 7,8,  9, 10, 11])
plt.savefig("plots/ablation_gnn_layers.pdf", dpi=300, bbox_inches = "tight")
plt.show()
plt.close()
plt.cla()
plt.clf()
all_perfs = []
for M in [0.01, 0.001, 0.5, 2.0]:
    all_perfs += [pd.DataFrame(get_last_performance_json(f"logs/GDSC2/ablation/smoothing_fold_{cv}_loss_kl_256_9_6_2.5_{M}_0.05_3_True.json")for cv in range(7)).assign(mse_weight = M)]

all_perfs += [complete_model.assign(mse_weight = 0.25)]
all_perfs = pd.concat(all_perfs)

p = sns.lineplot(x = "mse_weight", y=0, data = all_perfs.reset_index(), ci=95, err_style='bars')
p.set_ylabel("MSE")
p.set_xlabel("Weight $\lambda$ for MSE ")
p.set_xticks([0.001, 0.01, 0.25, 0.5, 2.0])
plt.vlines(0.25, 0, 0.0075, color = "red", linewidth = 1, linestyle = "--",)
p.set_ylim(0.0029, 0.0065)
p.set(xscale="log")
plt.savefig("plots/ablation_mse_weight.pdf", dpi=300, bbox_inches = "tight")
plt.show()
plt.close()
plt.cla()
plt.clf()
all_perfs = []
for T in [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]:
    all_perfs += [pd.DataFrame(get_last_performance_json(f"logs/GDSC2/ablation/smoothing_fold_{cv}_loss_kl_256_9_6_{T}_0.25_0.05_3_True.json")for cv in range(7)).assign(temperature = T)]

all_perfs += [complete_model.assign(temperature = 2.5)]
all_perfs = pd.concat(all_perfs)

p = sns.lineplot(x = "temperature", y=0, data = all_perfs.reset_index(), ci=95, err_style='bars')
p.set_ylabel("MSE")
p.set_xlabel("Temperature $\\tau$ using during softmax normalization")
plt.vlines(2.5, 0, 0.0075, color = "red", linewidth = 1, linestyle = "--")
p.set_ylim(0.0029, 0.0040)
p.set_xticks([5.0,3.0, 4.0, 2.5, 2.0, 1.5, 1.0])
plt.savefig("plots/ablation_temperature.pdf", dpi=300, bbox_inches = "tight")
plt.show()
plt.close()
plt.cla()
plt.clf()
no_gates_perf = pd.DataFrame(get_last_performance_json(f"logs/GDSC2/ablation/smoothing_fold_{cv}_loss_kl_256_9_6_2.5_0.25_0.05_3_False.json") for cv in range(7)).assign(learnable_gates = False)
performance_df = pd.concat([complete_model.assign(learnable_gates = True), no_gates_perf])

plotting_parameters = {
    'data':    performance_df,
    'x':       "learnable_gates",
    'y':       0,
    "orient":"v",
}
p = sns.boxplot(**plotting_parameters)
p.set_ylabel("MSE")
p.set_xlabel("Learnable gates for residual connections")
plt.savefig("plots/ablation_learnable_gates.pdf", dpi=300, bbox_inches = "tight")
plt.show()
plt.close()
plt.cla()
plt.clf()
# Create Drug discovery and precision oncology plots
dd_kl = [get_last_performance_json(f"logs/GDSC2/drug_discovery_fold_{cv}_loss_kl.json") for cv in range(10)]
dd_mse = [get_last_performance_json(f"logs/GDSC2/drug_discovery_fold_{cv}_loss_mse.json") for cv in range(10)]
po_mse = [get_last_performance_json(f"logs/GDSC2/precision_oncology_fold_{cv}_loss_mse.json") for cv in range(10)]
po_kl = [get_last_performance_json(f"logs/GDSC2/precision_oncology_fold_{cv}_loss_kl.json") for cv in range(10)]
funfor_dd = pd.read_csv("results/resultsfunfor_dd.csv", index_col=0)
funfor_po = pd.read_csv("results/resultsfunfor_po.csv", index_col=0)
funfor_dd.columns = ["MSE"]
funfor_po.columns = ["MSE"]
dd_perf = pd.concat([pd.DataFrame([dd_kl], index = ["MSE"]).T.assign(model = "ARCANet (KL)"),
          pd.DataFrame([dd_mse], index = ["MSE"]).T.assign(model = "ARCANet (MSE)"),
          funfor_dd.assign(model = "FunFor")])
po_perf = pd.concat([pd.DataFrame([po_kl], index = ["MSE"]).T.assign(model = "ARCANet (KL)"),
          pd.DataFrame([po_mse], index = ["MSE"]).T.assign(model = "ARCANet (MSE)"),
          funfor_dd.assign(model = "FunFor")])
plotting_parameters = {
    'data':    dd_perf,
    'x':       'MSE',
    'y':       'model',
    "palette": [palette[4], palette[3], palette[5]],
    "orient":"h",
}
pvalues = []
for model in ["ARCANet (MSE)", "FunFor"]:
    pvalues += [ttest_rel(dd_perf.query("model == 'ARCANet (KL)'").loc[:, "MSE"].to_numpy(),
          dd_perf.query("model == @model").loc[:, "MSE"].to_numpy())[1]]
p = sns.boxplot(**plotting_parameters)
pairs = [["ARCANet (KL)", "ARCANet (MSE)"],["ARCANet (KL)", "FunFor"]]
annotator = Annotator(p,pairs, **plotting_parameters)
annotator.set_pvalues(pvalues)
annotator.annotate()
p.set_ylabel("")
plt.savefig("plots/drug_discovery.pdf", dpi=300, bbox_inches = "tight")
plt.show()
plt.close()
plt.cla()
plt.clf()
plotting_parameters = {
    'data':    po_perf,
    'x':       'MSE',
    'y':       'model',
    "palette": [palette[4], palette[3], palette[5]],
    "orient":"h",
}
pvalues = []
for model in ["ARCANet (MSE)", "FunFor"]:
    pvalues += [ttest_rel(po_perf.query("model == 'ARCANet (KL)'").loc[:, "MSE"].to_numpy(),
          dd_perf.query("model == @model").loc[:, "MSE"].to_numpy())[1]]
p = sns.boxplot(**plotting_parameters)
pairs = [["ARCANet (KL)", "ARCANet (MSE)"],["ARCANet (KL)", "FunFor"]]
annotator = Annotator(p,pairs, **plotting_parameters)
annotator.set_pvalues(pvalues)
annotator.annotate()
p.set_ylabel("")
plt.savefig("plots/precision_oncology.pdf", dpi=300, bbox_inches = "tight")
plt.show()
plt.close()
plt.cla()
plt.clf()
# Create extrapolation plot
extra_kl = [get_last_performance_json(f"logs/GDSC2/extrapolation_fold_{cv}_loss_kl.json") for cv in range(10)]
extra_mse = [get_last_performance_json(f"logs/GDSC2/extrapolation_fold_{cv}_loss_mse.json") for cv in range(10)]
extra_r = pd.read_csv("results/extrap2pr.csv").to_numpy()[:, 1]
extra_3p = pd.read_csv("results/Fulllogistic_extrapolation.csv").to_numpy()[:, 1]
extra_4p = pd.read_csv("results/Fulllogistic2_extrapolation.csv").to_numpy()[:, 1]
all_perf = pd.DataFrame([extra_kl, extra_mse, extra_r, extra_3p, extra_4p]).T
models = ["ARCANet (KL)",
                    "ARCANet (MSE)",
                    "2P mixed-effect logistic model",
                    "3P fixed-effect logistic model",
                    "4P fixed-effect logistic model",]

all_perf.columns = models
all_perf = all_perf.unstack().reset_index().drop("level_1", axis=1)
all_perf.columns = ["model", "MSE"]
plotting_parameters = {
    'data':    all_perf,
    'x':       'MSE',
    'y':       'model',
    "palette": [palette[4], palette[3], palette[2], palette[1], palette[0]],
    "orient":"h",
}
pvalues = []
for model in models[1:]:
    pvalues += [ttest_rel(all_perf.query("model == 'ARCANet (KL)'").loc[:, "MSE"].to_numpy(),
          all_perf.query("model == @model").loc[:, "MSE"].to_numpy())[1]]
ax = plt.axes()
pairs = [[models[0], models[i]] for i in range(1, 5)]
g = sns.boxplot(ax=ax, **plotting_parameters)
annotator = Annotator(ax,pairs, **plotting_parameters)
annotator.set_pvalues(pvalues)
ax.set_ylabel("")
annotator.annotate()
plt.savefig("plots/extrapolation.pdf", dpi=300, bbox_inches="tight")
plt.show()