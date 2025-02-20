#!/bin/bash

if [ $# != 11 ]
then
    echo -e "usage:$0 [path to train_dir & d5.py] [first_time==1]"
    echo "$@"
    echo "$#"
    exit
fi

path=$1
first_time=$2
experiment_id=$3
constraints_id=$4
threshold=$5
max_actor_epochs=$6
x1=$7
x2=$8
lambda_=$9
original_model=${10}
snt_model_wo_ibp=${11}

##Bring up the learner:
if [ $first_time -eq 1 ];
then
    /users/`whoami`/venv/bin/python $path/d5.py --job_name=learner --task=0 --base_path=$path --experiment_id=${experiment_id} --constraints_id=${constraints_id} --threshold=${threshold} --max_actor_epochs=${max_actor_epochs} --x1=${x1} --x2=${x2} --lambda_=${lambda_} --original_model=${original_model} --snt_model_wo_ibp=${snt_model_wo_ibp} &
elif [ $first_time -eq 4 ]
then
    /users/`whoami`/venv/bin/python $path/d5.py --job_name=learner --task=0 --base_path=$path --load --eval --experiment_id=${experiment_id} --constraints_id=${constraints_id} --threshold=${threshold} --max_actor_epochs=${max_actor_epochs} --x1=${x1} --x2=${x2} --lambda_=${lambda_} --original_model=${original_model} --snt_model_wo_ibp=${snt_model_wo_ibp} &
else
    /users/`whoami`/venv/bin/python $path/d5.py --job_name=learner --task=0 --base_path=$path --load --experiment_id=${experiment_id} --constraints_id=${constraints_id} --threshold=${threshold} --max_actor_epochs=${max_actor_epochs} --x1=${x1} --x2=${x2} --lambda_=${lambda_} --original_model=${original_model} --snt_model_wo_ibp=${snt_model_wo_ibp} &
fi
