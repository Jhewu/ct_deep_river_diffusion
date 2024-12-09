#!/bin/bash

# This is a bash file for running 
# diffusion model hyperparameter 
# test. 

#----------PARAMETERS----------
folder_paths=("exp10" "exp11" "exp12" "exp13" "exp14" "exp15" "exp16")
train_model_list=(True)
load_and_train_list=(False)
image_sizes=("200,600" "200,600" "200,600" "200,600" "200,600" "200,600" "200,600")
num_epochs_list=(100)
batch_sizes=(4 4 4 4 4 4 4)
learning_rates=(3e-4 3e-3 3e-4 3e-4 3e-4 3e-4 3e-4)
weight_decays=(1e-4)
kid_image_sizes=(75)
plot_diffusion_steps_list=(20)
kid_diffusion_steps_list=(5)
min_signal_rates=(0.02 0.02 0.20 0.40 0.02 0.15 0.25)
max_signal_rates=(0.95 0.10 0.80 0.50 0.99 0.95 0.75)
embedding_dims_list=(32)
widths_list=("32, 64, 96, 128" "32, 64, 96, 128" "32, 64, 96, 128" "32, 64, 96, 128" "32, 64, 96, 128" "32, 64, 96, 128" "32, 64, 96, 128")
block_depth_list=(2 2 2 2 2 2 2)
checkpoint_monitor_list=("val_kid")
early_stop_monitor_list=("val_kid")
early_stop_min_delta_list=(3e-5 3e-5 3e-5 3e-5 3e-5 3e-5 3e-5)
early_stop_patience_list=(25)
early_stop_start_epoch_list=(50)
images_to_generate_list=(5)
generate_diffusion_steps_list=(50)

#----------RUNTIME----------
for i in ${!folder_paths[@]}
do
    # Modify paramters.py 
    sed -i "s/folder_path = .*/folder_path = \"${folder_paths[i]}\"/" parameters.py
    sed -i "s/train_model = .*/train_model = ${train_model_list[0]}/" parameters.py
    sed -i "s/load_and_train = .*/load_and_train = ${load_and_train_list[0]}/" parameters.py
    sed -i "s/image_size = .*/image_size = (${image_sizes[i]})/" parameters.py
    sed -i "s/num_epochs = .*/num_epochs = ${num_epochs_list[0]}/" parameters.py
    sed -i "s/batch_size = .*/batch_size = ${batch_sizes[i]}/" parameters.py
    sed -i "s/learning_rate = .*/learning_rate = ${learning_rates[i]}/" parameters.py
    sed -i "s/weight_decay = .*/weight_decay = ${weight_decays[0]}/" parameters.py
    sed -i "s/kid_image_size = .*/kid_image_size = ${kid_image_sizes[0]}/" parameters.py
    sed -i "s/plot_diffusion_steps = .*/plot_diffusion_steps = ${plot_diffusion_steps_list[0]}/" parameters.py
    sed -i "s/kid_diffusion_steps = .*/kid_diffusion_steps = ${kid_diffusion_steps_list[0]}/" parameters.py
    sed -i "s/min_signal_rate = .*/min_signal_rate = ${min_signal_rates[i]}/" parameters.py
    sed -i "s/max_signal_rate = .*/max_signal_rate = ${max_signal_rates[i]}/" parameters.py
    sed -i "s/embedding_dims = .*/embedding_dims = ${embedding_dims_list[0]}/" parameters.py
    sed -i "s/widths = .*/widths = [${widths_list[i]}]/" parameters.py
    sed -i "s/block_depth = .*/block_depth = ${block_depth_list[i]}/" parameters.py
    sed -i "s/checkpoint_monitor = .*/checkpoint_monitor = \"${checkpoint_monitor_list[0]}\"/" parameters.py
    sed -i "s/early_stop_monitor = .*/early_stop_monitor = \"${early_stop_monitor_list[0]}\"/" parameters.py
    sed -i "s/early_stop_min_delta = .*/early_stop_min_delta = ${early_stop_min_delta_list[i]}/" parameters.py
    sed -i "s/early_stop_patience = .*/early_stop_patience = ${early_stop_patience_list[0]}/" parameters.py
    sed -i "s/early_stop_start_epoch = .*/early_stop_start_epoch = ${early_stop_start_epoch_list[0]}/" parameters.py
    sed -i "s/images_to_generate = .*/images_to_generate = ${images_to_generate_list[0]}/" parameters.py
    sed -i "s/generate_diffusion_steps = .*/generate_diffusion_steps = ${generate_diffusion_steps_list[0]}/" parameters.py

    # Run your main script
    python ddim.py
done







