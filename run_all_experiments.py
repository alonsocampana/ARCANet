import argparse

tasks = []
# Hyperparameter optimization
tasks.append(f"python3 optimize_hyperparameters.py --dataset GDSC1 --setting precision_oncology --fold 0 --max_iter 100 ")
tasks.append(f"python3 optimize_hyperparameters.py --dataset NCI60 --setting drug_discovery --leave_out 1 --fold 0 --max_iter 25 ")

############################################
# main experiments
for dataset in ["CTRPv2", "PRISM", "GDSC2"]:
    for setting in ["smoothing", "extrapolation", "interpolation", "precision_oncology"]:
        for fold in range(10):
            tasks.append(f"python3 train_model.py --dataset {dataset} " + 
            f"--dataset_hyperpar GDSC1 --setting {setting} "
            f"--setting_hyperpar precision_oncology  --fold {fold} "+
            f"--max_iter 100")
            tasks.append(f"python3 train_model.py --dataset {dataset} " + 
            f"--dataset_hyperpar GDSC1 --setting {setting} "
            f"--setting_hyperpar precision_oncology  --fold {fold} "+
            f"--max_iter 100 --fingerprint ")
            
for bootstrap in range(10):
    for dataset in ["NCI60"]:
        for setting in ["smoothing", "extrapolation", "interpolation"]:
            for fold in range(1, 2):
                tasks.append(f"python3 train_model.py --dataset {dataset} " + 
                f"--dataset_hyperpar NCI60 --setting {setting} "
                f"--setting_hyperpar drug_discovery  --fold {fold} "+
                f"--max_iter 25 --fingerprint --random_suffix ")
                tasks.append(f"python3 train_model.py --dataset {dataset} " + 
                f"--dataset_hyperpar NCI60 --setting {setting} "
                f"--setting_hyperpar drug_discovery  --fold {fold} "+
                f"--max_iter 25 --random_suffix ")
    for dataset in ["NCI60"]:
        for setting in ["drug_discovery"]:
            for fold in range(1, 2):
                tasks.append(f"python3 train_model.py --dataset {dataset} " + 
                f"--dataset_hyperpar NCI60 --setting {setting} "
                f"--setting_hyperpar drug_discovery  --fold {fold} "+
                f"--max_iter 25 --fingerprint --random_suffix ")
                tasks.append(f"python3 train_model.py --dataset {dataset} " + 
                f"--dataset_hyperpar NCI60 --setting {setting} "
                f"--setting_hyperpar drug_discovery  --fold {fold} "+
                f"--max_iter 25 --random_suffix ")
##########################################
# baselines (main experiment)
for logistic in ["2P", "3P", "4P"]:
    for dataset in ["GDSC2", "CTRPv2", "PRISM"]:
        for setting in ["smoothing", "extrapolation", "interpolation"]:
            for fold in range(10):
                tasks.append(f"python3 train_baseline_individual.py --dataset {dataset} --setting {setting} --fold {fold} --logistic {logistic} ")
    for dataset in ["NCI60"]:
        for setting in ["smoothing", "extrapolation", "interpolation"]:
            for fold in range(10):
                tasks.append(f"python3 train_baseline_individual.py --dataset {dataset} --setting {setting} --fold {fold} --logistic {logistic} ")
    for dataset in ["NCI60"]:
        for setting in ["smoothing", "extrapolation", "interpolation"]:
            for fold in range(10):
                tasks.append(f"python3 train_baseline_individual.py --dataset {dataset} --setting {setting} --fold {fold} --cuda 0 --logistic {logistic}")
for dataset in ["GDSC2", "CTRPv2", "PRISM"]:
    for setting in ["precision_oncology"]:
        for fold in range(10):
            tasks.append(f"python3 train_funfor.py --dataset {dataset} --setting {setting} --fold {fold} ")
for dataset in ["NCI60"]:
    for setting in ["drug_discovery"]:
        for fold in range(10):
            tasks.append(f"python3 train_funfor.py --dataset {dataset} --setting {setting} --fold {fold} ")

for dataset in ["GDSC2", "CTRPv2", "PRISM", "NCI60"]:
    for setting in ["smoothing", "extrapolation", "interpolation"]:
        for fold in range(10):
            tasks.append(f"python3 train_baseline.py --dataset {dataset} --setting {setting} --fold {fold} --mixed_effect ")


##########################################
# ablation study

for dataset in ["PRISM"]:
    for setting in ["smoothing", "precision_oncology"]:
        for fold in range(10):
            tasks.append(f"python3 train_model.py --dataset {dataset} " + 
            f"--dataset_hyperpar GDSC1 --setting {setting} "
            f"--setting_hyperpar precision_oncology  --fold {fold} "+
            f"--max_iter 100 --linear_head ")
            tasks.append(f"python3 train_model.py --dataset {dataset} " + 
            f"--dataset_hyperpar GDSC1 --setting {setting} "
            f"--setting_hyperpar precision_oncology  --fold {fold} "+
            f"--max_iter 100 --nokl ")
            tasks.append(f"python3 train_model.py --dataset {dataset} " + 
            f"--dataset_hyperpar GDSC1 --setting {setting} "
            f"--setting_hyperpar precision_oncology  --fold {fold} "+
            f"--max_iter 100 --fingerprint --linear_head ")
            tasks.append(f"python3 train_model.py --dataset {dataset} " + 
            f"--dataset_hyperpar GDSC1 --setting {setting} "
            f"--setting_hyperpar precision_oncology  --fold {fold} "+
            f"--max_iter 100 --fingerprint --nokl ")
            tasks.append(f"python3 train_model.py --dataset {dataset} " + 
            f"--dataset_hyperpar GDSC1 --setting {setting} "
            f"--setting_hyperpar precision_oncology  --fold {fold} "+
            f"--max_iter 100 --fusion_concat ")
            tasks.append(f"python3 train_model.py --dataset {dataset} " + 
            f"--dataset_hyperpar GDSC1 --setting {setting} "
            f"--setting_hyperpar precision_oncology  --fold {fold} "+
            f"--max_iter 100 --fusion_concat --fingerprint ")
            
##########################################            
# Missing data experiments

for dataset in ["PRISM"]:
    for setting in ["smoothing", "precision_oncology"]:
        for missing in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.995, 0.9975, 0.99875]:
            for fold in range(10):
                tasks.append(f"python3 train_model.py --dataset {dataset} " + 
                f"--dataset_hyperpar GDSC1 --setting {setting} "
                f"--setting_hyperpar precision_oncology  --fold {fold} "+
                f"--max_iter 100 --missing_random {missing} --fingerprint ")
                for sigmoid in ["2P", "3P", "4P"]:
                    tasks.append(f"python3 train_baseline_individual.py --dataset {dataset} --setting {setting} --fold {fold} --logistic {logistic} --missing_random {missing} ")
##########################################
parser = argparse.ArgumentParser(description="Train sigmoid baseline")
parser.add_argument(
    "--cuda",
    type=int,
    default = 0,
    required=False,
    help="Cuda device to perform the experiments"
)
args= parser.parse_args()
cuda_n = args.cuda
for experiment in tasks:
    try:
        os.system(f"{experiment} --cuda {cuda_n}")
    except Exception as e:
        print(e)