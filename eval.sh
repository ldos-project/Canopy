if [ $# -eq 18 ]
then
    echo "C3: $C3_TRACE_DIR"
    source setup.sh

    first_time=$1
    port_base=$2
    model_name=$3
    constraints_id=$4
    threshold=$5 # For evaluation.
    max_actor_epochs=$6
    x1=$7
    x2=$8
    lambda_=$9
    original_model=${10}
    snt_model_wo_ibp=${11}
    cur_dir=`pwd -P`
    scheme_="cubic"
    max_steps=500000         #Run untill you collect 50k samples per actor
    eval_duration=300000000
    num_actors=${12}
    k_symbolic_components=${13} # For evaluation.
    k=${14} # For evaluation.
    trace_name=${15}
    reward_mode=${16}
    seed=${17}
    bdp_multiplier=${18}

    memory_size=$((max_steps*$num_actors))
    dir="${cur_dir}/rl-module"

    if [ $model_name == "tcp-cubic" ]
    then
        only_tcp=1
    else
        only_tcp=0
    fi

    sed "s/\"ckptdir\"\: \"\"/\"ckptdir\"\: \"$model_name\"/" $cur_dir/params_base_eval.json > "${dir}/params.json"
    sed -i "s/\"memsize\"\: 2553600/\"memsize\"\: $memory_size/" "${dir}/params.json"
    sudo killall -s9 python client orca-server-mahimahi_v2

    epoch=20
    act_port=$port_base

    # If you are here: You are going to perform an evaluation over an emulated link

    echo "./learner_v2.sh  $dir"
    ./learner_v2.sh  $dir \
        ${first_time} \
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
        ${seed} &
    #Bring up the actors:
    act_id=0
    downl="$trace_name"
    echo "downl=$downl"
    upl="wired192"

    del=10

    # Check if $downl has a bandwidth.
    if [[ $downl == *"wired"* ]]
    then
        dl=`echo $downl | grep -oP 'wired[0-9]*\.?[0-9]+' | sed 's/wired//g'`
    elif [[ $downl == *"bump"* ]] || [[ $downl == *"sawtooth"* ]] || [[ $downl == *"mountain"* ]]
    then
        dl=`echo $downl | grep -oP '\d+$'`
    else
        dl=30 # Most time steps are less than 30Mbps in the variable link traces.
    fi

    # Use qsize = bdp_multiplier*BDP
    bdp=$(echo "2 * $dl * $del / 12.0" | bc -l)
    qs=$(echo "$bdp_multiplier * $bdp" | bc -l | awk '{print int($1)}')

    echo "Start actor with actor_id=$act_id"
    ./actor_v2.sh \
        ${act_port} \
        $epoch \
        ${first_time} \
        $scheme_ \
        $dir \
        $act_id \
        $downl \
        $upl \
        $del \
        $eval_duration \
        $qs \
        $max_steps \
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
        ${seed} &
    pids="$pids $!"
    for pid in $pids
    do
        echo "waiting for $pid"
        wait $pid
    done

    #Bring down the learner and actors ...
    for i in `seq 0 $((num_actors))`
    do
        sudo killall -s15 python
        sudo killall -s15 orca-server-mahimahi_v2
        sudo killall -s15 client
    done
    sudo killall -s9 python client orca-server-mahimahi_v2
else
    echo "usage: $0 [{Learning from scratch=1} {Continue your learning=0} {Just Do Evaluation=4}] [base port number ] ... (18 args)"
    echo "\t Error found only $# args. Expected 17"
fi

# Convert params back.
# sed $cur_dir/params_base.json > "${dir}/params.json"
# Make sure all are down ...
sudo killall -s9 python client orca-server-mahimahi_v2
