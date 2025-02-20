#!/bin/bash

# This experiments set the training reward as raw reward + symbolic constraint
# (1 - lambda_) * raw reward + lambda_ * symbolic reward
# Default Parameters
num_actors=1
port_base=44444
constraints_id=6
max_actor_epochs=10000
original_model=0
snt_model_wo_ibp=0
# threshold=0.25 # As the input, it should threshold * 100
x1=5
# x2=15

# Input parameters
lambda_=$1 # The weight before R_certified.
k_symbolic_components=$2 # Number of symbolic components to use.
k=$3 # The window for the property.
reward_mode=$4 # by default "raw-sym"
seed=$5 # The seed for the random number generator.
constraints_id=$6 # The id of the constraint to use. 6 for safety; 7 for robustness.
# The final threshold is input_threshold / 100.
# In safety, threshold is the cutoff for small inverseRTT; 1-threshold is the cutoff for large inverseRTT.
# In robustness, threshold is the noise range. e.g. input_threshold=5 -> threshold=0.05 -> noise is from [-5%, 5%]. 
input_threshold=$7 
# x2 is the expected output range.
# In safety, x2 is the cutoff for cwnd change (cwnd_i - cwnd_{i-1}).
# In robustness, x2 is the cutoff for the noised cwnd change (cwnd_noise_i - cwnd_i).
x2=$8

threshold_base=100.0
threshold=$(bc <<< "scale=2 ; $input_threshold/$threshold_base")

orca_lambda=0.0
if [ $(echo "$lambda_ == $orca_lambda" | bc) -eq 1 ]
then
    model_name="orca-"${num_actors}"actors"
else
    model_name="c3-"${num_actors}"actors-"${seed}"-"${constraints_id}"-threshold"${threshold}"-x2"${x2}
fi

echo "Model name: ${model_name}"

# Run the training for 5 sessions.
./orca_v2_multi.sh 1 ${port_base} ${model_name} ${constraints_id} ${threshold} ${max_actor_epochs} ${x1} ${x2} ${lambda_} ${original_model} ${snt_model_wo_ibp} ${num_actors} ${k_symbolic_components} ${k} ${reward_mode} ${seed}
echo "First time done"
./orca_v2_multi.sh 2 ${port_base} ${model_name} ${constraints_id} ${threshold} ${max_actor_epochs} ${x1} ${x2} ${lambda_} ${original_model} ${snt_model_wo_ibp} ${num_actors} ${k_symbolic_components} ${k} ${reward_mode} ${seed}
echo "Second time done"
# ./orca_v2_multi.sh 2 ${port_base} ${model_name} ${constraints_id} ${threshold} ${max_actor_epochs} ${x1} ${x2} ${lambda_} ${original_model} ${snt_model_wo_ibp} ${num_actors} ${k_symbolic_components} ${k} ${reward_mode} ${seed}
# echo "Third time done"
# ./orca_v2_multi.sh 2 ${port_base} ${model_name} ${constraints_id} ${threshold} ${max_actor_epochs} ${x1} ${x2} ${lambda_} ${original_model} ${snt_model_wo_ibp} ${num_actors} ${k_symbolic_components} ${k} ${reward_mode} ${seed}
# echo "Fourth time done"
# ./orca_v2_multi.sh 2 ${port_base} ${model_name} ${constraints_id} ${threshold} ${max_actor_epochs} ${x1} ${x2} ${lambda_} ${original_model} ${snt_model_wo_ibp} ${num_actors} ${k_symbolic_components} ${k} ${reward_mode} ${seed}
# echo "Fifth time done"
