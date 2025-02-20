#!/bin/bash
if [ $# -ne 27 ] && [ $# -ne 28 ]; then
    echo "Expected 27 or 28 args but got only $# args"
    echo -e "usage:$0 port period first_time [underlying scheme:cubic , vegas , westwood , illinois , bbr, yeah , veno, scal , htcp , cdg , hybla ,... ] [path to ddpg.py] [actor id] [downlink] [uplink] [one-way link delay] [time time] [Qsize] [Max iterations per run]"
    exit
fi

echo "in actor_v2.sh"

if [ $# -eq 27 ] || [ $# -eq 28 ]
then
    port=$1
    period=$2
    first_time=$3
    scheme=$4
    path=$5
    id=$6
    down=$7
    up=$8
    latency=$9
    finish_time=${10}
    qsize=${11}
    max_it=${12}
    model_name=${13}
    constraints_id=${14}
    threshold=${15}
    max_actor_epochs=${16}
    x1=${17}
    x2=${18}
    lambda_=${19}
    original_model=${20}
    snt_model_wo_ibp=${21}
    k_symbolic_components=${22} 
    k=${23}
    only_tcp=${24}
    trace_name=${25}
    reward_mode=${26}
    seed=${27}
    if [ $# -eq 28 ]; then
        training_session_idx=${28}
    else
        echo "[actor_v2.sh] training_session_idx was not set when this script was called"
        training_session_idx=-1
    fi
else
    echo "[Error] args for actor_v2.sh ill formed. Fatal error."
    exit 1
fi
echo "Running orca-$scheme: $down"

trace=""
scheme_des="orca-$scheme-$latency-$period-$qsize-${model_name}-${constraints_id}-${threshold}"
log="orca-$scheme-$down-$up-$latency-${period}-$qsize-${model_name}-${constraints_id}-${threshold}-${seed}"

#Bring up the actor i:
echo "will be done in $finish_time seconds ..."
echo "$path/orca-server-mahimahi $port $path ${period} ${first_time} $scheme $id $down $up $latency $log $finish_time $qsize $max_it ${model_name} ${constraints_id} ${threshold} ${max_actor_epochs} ${x1} ${x2} ${lambda_} ${original_model} ${snt_model_wo_ibp} ${k_symbolic_components} ${k} ${only_tcp}"

$path/orca-server-mahimahi_v2 \
    $port \
    $path \
    ${period} \
    ${first_time} \
    $scheme \
    $id \
    $down \
    $up \
    $latency \
    $log \
    $finish_time \
    $qsize \
    $max_it \
    ${model_name} \
    ${constraints_id} \
    ${threshold} \
    ${max_actor_epochs} \
    ${x1} \
    ${x2} \
    ${lambda_} \
    ${original_model} \
    ${snt_model_wo_ibp} \
    ${k_symbolic_components} \
    ${k} \
    ${only_tcp} \
    ${trace_name} \
    ${reward_mode} \
    ${seed} \
    ${training_session_idx}

#sudo killall -s15 python
#sleep 10
echo "Finished. actor $id$"
# Only plot network bw when load & eval & no-learning.
if [ ${first_time} -eq 4 ]
then
    echo "Doing Some Analysis ..."
    out="sum-${log}.tr"
    echo $log >> /mydata/log/$out
    suffix="${trace_name}-${k_symbolic_components}-${k}"
    $path/mm-thr 500 /mydata/log/down-${log} 1>${model_name}-tmp-${suffix} 2>res_${model_name}-tmp-${suffix}
    cat res_${model_name}-tmp-${suffix} >>/mydata/log/$out
    mkdir -p "traffic_evaluation_figures/seed${seed}"
    mv "${model_name}-tmp-${suffix}" "traffic_evaluation_figures/seed${seed}/${model_name}-${suffix}.svg"
    rm res_${model_name}-tmp-${suffix}
fi
echo "Done"

