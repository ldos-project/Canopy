#!/bin/bash
if [ $# != 21 ]
then
    echo -e "usage:$0 port period first_time [underlying scheme:cubic , vegas , westwood , illinois , bbr, yeah , veno, scal , htcp , cdg , hybla ,... ] [path to ddpg.py] [actor id] [downlink] [uplink] [one-way link delay] [time time] [Qsize] [Max iterations per run]"
    exit
fi

port=$1
period=$2
first_time=$3
x=100
scheme=$4
path=$5
id=$6
down=$7
up=$8
latency=$9
finish_time=${10}
qsize=${11}
max_it=${12}
experiment_id=${13}
constraints_id=${14}
threshold=${15}
max_actor_epochs=${16}
x1=${17}
x2=${18}
lambda_=${19}
original_model=${20}
snt_model_wo_ibp=${21}

echo "Running orca-$scheme: $down"

trace=""
scheme_des="orca-$scheme-$latency-$period-$qsize-${experiment_id}-${constraints_id}-${threshold}"
log="orca-$scheme-$down-$up-$latency-${period}-$qsize-${experiment_id}-${constraints_id}-${threshold}"

#Bring up the actor i:
echo "will be done in $finish_time seconds ..."
echo "$path/orca-server-mahimahi $port $path ${period} ${first_time} $scheme $id $down $up $latency $log $finish_time $qsize $max_it ${experiment_id} ${constraints_id} ${threshold} ${max_actor_epochs} ${x1} ${x2} ${lambda_} ${original_model} ${snt_model_wo_ibp}"

$path/orca-server-mahimahi $port $path ${period} ${first_time} $scheme $id $down $up $latency $log $finish_time $qsize $max_it ${experiment_id} ${constraints_id} ${threshold} ${max_actor_epochs} ${x1} ${x2} ${lambda_} ${original_model} ${snt_model_wo_ibp} 

#sudo killall -s15 python
#sleep 10
echo "Finished. actor $id$"
if [ ${first_time} -eq 1 ] || [ ${first_time} -eq 2 ] || [ ${first_time} -eq 4 ]
then
    echo "Doing Some Analysis ..."
    out="sum-${log}.tr"
    echo $log >> $path/log/$out
    $path/mm-thr 500 $path/log/down-${log} 1>tmp-${id}-${experiment_id}-${constraints_id}-${threshold}-${max_actor_epochs}-${x1}-${x2}-${lambda_}-${original_model}-${snt_model_wo_ibp} 2>res_tmp-${id}-${experiment_id}-${constraints_id}-${threshold}-${max_actor_epochs}-${x1}-${x2}-${lambda_}-${original_model}-${snt_model_wo_ibp}
    cat res_tmp-${id}-${experiment_id}-${constraints_id}-${threshold}-${max_actor_epochs}-${x1}-${x2}-${lambda_}-${original_model}-${snt_model_wo_ibp} >>$path/log/$out
    echo "------------------------------" >> $path/log/$out
    if [ ${first_time} -eq 1 ] || [ ${first_time} -eq 2 ]
    then
        mv "tmp-$id-${experiment_id}-${constraints_id}-${threshold}-${max_actor_epochs}-${x1}-${x2}-${lambda_}-${original_model}-${snt_model_wo_ibp}" "traffic_figures/tmp-$id-${experiment_id}-${constraints_id}-${threshold}-${max_actor_epochs}-${x1}-${x2}-${lambda_}-${original_model}-${snt_model_wo_ibp}.svg"
    fi
    if [ ${first_time} -eq 4 ]
    then
        mv "tmp-$id-${experiment_id}-${constraints_id}-${threshold}-${max_actor_epochs}-${x1}-${x2}-${lambda_}-${original_model}-${snt_model_wo_ibp}" "traffic_evaluation_figures/tmp-$id-${experiment_id}-${constraints_id}-${threshold}-${max_actor_epochs}-${x1}-${x2}-${lambda_}-${original_model}-${snt_model_wo_ibp}.svg"
    fi
    rm res_tmp-${id}-${experiment_id}-${constraints_id}-${threshold}-${max_actor_epochs}-${x1}-${x2}-${lambda_}-${original_model}-${snt_model_wo_ibp}
    rm $path/log/$out
    rm $path/log/down-${log}
fi
echo "Done"

