# Prepare data and folder structure

python3 prepare_folder.py
python3 prepara_data.py

# Total number of DL subexperiments: 445
# Run main experiments
# Run smoothing experiments for different amounts of data, all folds using both losses
for d in {1..4}
do
    for f in {0..6}
    do
       python3 fit_curves_DL.py --setting smoothing --fold $f --dropped $d --loss mse
       python3 fit_curves_DL.py --setting smoothing --fold $f --dropped $d --loss kl
    done
done
# Run interpolation experiments for different amounts of data, all folds using both losses
for d in {1..4}
do
    for f in {0..4}
    do
       python3 fit_curves_DL.py --setting interpolation --fold $f --dropped $d --loss mse
       python3 fit_curves_DL.py --setting interpolation --fold $f --dropped $d --loss kl
    done
done
# Run extrapolation experiments for all folds using both losses
for f in {0..4}
do
   python3 fit_curves_DL_extrapolation.py --fold $f --loss kl
   python3 fit_curves_DL_extrapolation.py --fold $f --loss mse
done
# Run experiments for drug-discovery and precision oncology using both losses
for f in {0..9}
do
   python fit_curves_DL_blinds.py --setting precision_oncology --fold $f --loss mse
   python fit_curves_DL_blinds.py --setting precision_oncology --fold $f --loss kl
   python fit_curves_DL_blinds.py --setting drug_discovery --fold $f --loss mse
   python fit_curves_DL_blinds.py --setting drug_discovery --fold $f --loss kl
done
# Run experiments for shrinked data
# Run experiments for randomly and systematically decreased training sizes
for d in 1 2 4 6 8
do
    for f in {0..6}
    do
        python3 fit_curves_DL.py --setting smoothing --fold $f --dropped 1 --loss kl --drop_random $d
        python3 fit_curves_DL.py --setting smoothing --fold $f --dropped 1 --loss kl --drop_classes $d
    done
done
# Run drug discovery experiments with lower number of drugs
do
for d in 1 3 5 7
do
    for f in {0..9}
    do
        python fit_curves_DL_blinds.py --setting drug_discovery --fold $f --loss kl --drop_drugs $d
    done
done
# Run ablation experiments
# Run experiments for a longer number of epochs
for f in {0..6}
do
    python3 fit_curves_DL.py --setting smoothing --fold $f --dropped 1 --loss mse --max_epoch 300
done
#different embedding sizes
for e in 64 128 512 1024
    for f in {0..6}
    do
        python3 fit_curves_DL_ablation.py --setting smoothing --fold $f --dropped 1 --loss kl --embed_size $e
    done
done
#different number of GAT layers
for l in 1 2 5 7 8 10 11
    for f in {0..6}
    do
        python3 fit_curves_DL_ablation.py --setting smoothing --fold $f --dropped 1 --loss kl --gat_layers $l
    done
done
#different MSE weights
for w in 0 0.001 0.01 0.5 2
    for f in {0..6}
    do
        python3 fit_curves_DL_ablation.py --setting smoothing --fold $f --dropped 1 --loss kl --weight_mse $w
    done
done
#different temperatures
for t in 0.1 1.0 1.5 2.0 3.0 4.0 5.0 10.0
    for f in {0..6}
    do
        python3 fit_curves_DL_ablation.py --setting smoothing --fold $f --dropped 1 --loss kl --weight_mse $w
    done
done
#run model without learnable gates and without gaussian noise
for f in {0..6}
do
    python3 fit_curves_DL_ablation.py --setting smoothing --fold $f --dropped 1 --loss kl --no_gates
    python3 fit_curves_DL_ablation.py --setting smoothing --fold $f --dropped 1 --loss kl --gaussian_noise 0
done


# Run R baselines
for d in {1..4}
do
    python3 fit_curves_R.py --setting smoothing --dropped 1
    python3 fit_curves_R.py --setting interpolation --dropped 1
done
for d in 1 2 4 6 8
do
    python3 fit_curves_R.py --setting smoothing --dropped 1 --drop_classes $d
    python3 fit_curves_R.py --setting smoothing --dropped 1 --drop_classes $d
done
# FunFor
python3 FunFor/run_all_missing.py
python3 FunFor/run_all_folds.py

# Generate plots
python3 generate_plots.py