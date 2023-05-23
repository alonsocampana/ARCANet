import os

folders = ["plots", "logs", "results", "data", "params", "models", "logs/GDSC1", "logs/GDSC2", "logs/GDSC2/ablation", ]
for f in folders:
    if not os.path.exists(f):
        os.mkdir(f)