#!/bin/bash

if [ $# != 17 ]
then
    echo -e "usage:$0 [path to train_dir & d5.py] [first_time==1]"
    echo "$@"
    echo "$#"
    exit
fi

path=$1
first_time=$2
model_name=$3
constraints_id=$4
threshold=$5
max_actor_epochs=$6
x1=$7
x2=$8
lambda_=$9
original_model=${10}
snt_model_wo_ibp=${11}
k_symbolic_components=${12}
k=${13}
only_tcp=${14}
trace_name=${15}
reward_mode=${16}
seed=${17}

##Bring up the learner:
if [ $first_time -eq 1 ];
then
    /users/`whoami`/venv/bin/python $path/d5_v2.py \
        --job_name=learner \
        --task=0 \
        --base_path=$path \
        --model_name=${model_name} \
        --constraints_id=${constraints_id} \
        --threshold=${threshold} \
        --max_actor_epochs=${max_actor_epochs} \
        --x1=${x1} \
        --x2=${x2} \
        --lambda_=${lambda_} \
        --original_model=${original_model} \
        --snt_model_wo_ibp=${snt_model_wo_ibp} \
        --k_symbolic_components=${k_symbolic_components} \
        --k=${k} \
        --only_tcp=${only_tcp} \
        --trace_name=${trace_name} \
        --reward_mode=${reward_mode} \
        --seed=${seed} &
elif [ $first_time -eq 4 ]
then
    /users/`whoami`/venv/bin/python $path/d5_v2.py \
        --job_name=learner \
        --task=0 \
        --base_path=$path \
        --load \
        --eval \
        --model_name=${model_name} \
        --constraints_id=${constraints_id} \
        --threshold=${threshold} \
        --max_actor_epochs=${max_actor_epochs} \
        --x1=${x1} \
        --x2=${x2} \
        --lambda_=${lambda_} \
        --original_model=${original_model} \
        --snt_model_wo_ibp=${snt_model_wo_ibp} \
        --k_symbolic_components=${k_symbolic_components} \
        --k=${k} \
        --only_tcp=${only_tcp} \
        --trace_name=${trace_name} \
        --reward_mode=${reward_mode} \
        --seed=${seed} &
else
    /users/`whoami`/venv/bin/python $path/d5_v2.py \
        --job_name=learner \
        --task=0 \
        --base_path=$path \
        --load \
        --model_name=${model_name} \
        --constraints_id=${constraints_id} \
        --threshold=${threshold} \
        --max_actor_epochs=${max_actor_epochs} \
        --x1=${x1} \
        --x2=${x2} \
        --lambda_=${lambda_} \
        --original_model=${original_model} \
        --snt_model_wo_ibp=${snt_model_wo_ibp} \
        --k_symbolic_components=${k_symbolic_components} \
        --k=${k} \
        --only_tcp=${only_tcp} \
        --trace_name=${trace_name} \
        --reward_mode=${reward_mode} \
        --seed=${seed} &
fi
