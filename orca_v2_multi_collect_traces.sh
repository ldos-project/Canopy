if [ $# -eq 6 ]
then
    source setup.sh

    first_time=$1
    port_base=$2
    experiment_id=$3
    constraints_id=16
    threshold=0.5
    max_actor_epochs=$4
    x1=5
    x2=15
    lambda_=0.5
    original_model=0
    snt_model_wo_ibp=0
    cur_dir=`pwd -P`
    scheme_="cubic"
    max_steps=500000         #Run untill you collect 50k samples per actor
    eval_duration=30
    num_actors=$5
    k_symbolic_components=1
    k=1
    first_bw=$6
    memory_size=$((max_steps*$num_actors))
    dir="${cur_dir}/rl-module"

    bw_array=($(seq ${first_bw} 6 192))

    sed "s/\"num_actors\"\: 1/\"num_actors\"\: $num_actors/" $cur_dir/params_base.json > "${dir}/params.json"
    sed -i "s/\"memsize\"\: 2553600/\"memsize\"\: $memory_size/" "${dir}/params.json"
    sudo killall -s9 python client orca-server-mahimahi_v0 orca-server-mahimahi_v2

    epoch=20
    act_port=$port_base

    echo "First bw: $first_bw"

    # If you are here: You are going to start/continue learning a better model!

    #Bring up the learner:
    echo "./learner_v2.sh  $dir $first_time $experiment_id ${constraints_id} ${threshold} &"
    if [ $1 -eq 1 ];
    then
        # Start the learning from the scratch
        /users/`logname`/venv/bin/python ${dir}/d5_v2.py --job_name=learner --task=0 --base_path=${dir} --experiment_id=${experiment_id} --constraints_id=${constraints_id} --threshold=${threshold} --max_actor_epochs=${max_actor_epochs} --x1=${x1} --x2=${x2} --lambda_=${lambda_} --original_model=${original_model} --snt_model_wo_ibp=${snt_model_wo_ibp} --k_symbolic_components=${k_symbolic_components} --k=${k} --only_tcp=1&
        lpid=$!
    else
        # Continue the learning on top of previous model
        /users/`logname`/venv/bin/python ${dir}/d5_v2.py --job_name=learner --task=0 --base_path=${dir} --load --experiment_id=${experiment_id} --constraints_id=${constraints_id} --threshold=${threshold} --max_actor_epochs=${max_actor_epochs} --x1=${x1} --x2=${x2} --lambda_=${lambda_} --original_model=${original_model} --snt_model_wo_ibp=${snt_model_wo_ibp} --k_symbolic_components=${k_symbolic_components} --k=${k} --only_tcp=1&
        lpid=$!
    fi
    sleep 10

    #Bring up the actors:
    # Here, we go with single actor
    act_id=0
    # for dl in 48
    for dl in "${bw_array[@]}"
    do  
        if [ $first_bw == $dl ]
        then
            downl="wired${first_bw}"
        else
            if [ $dl -lt $first_bw ]
            then
                continue
            else
                downl="wired${first_bw}-$dl"
            fi
        fi
        echo "downl=$downl"
        upl=$downl
        for del in 10 # minRTT is 2*del
        do
            bdp=$((2*dl*del/12))      #12Mbps=1pkt per 1 ms ==> BDP=2*del*BW=2*del*dl/12
            for qs in $((2*bdp))
            do
                echo "Start actor $act_id"
                ./actor_v2.sh ${act_port} $epoch ${first_time} $scheme_ $dir $act_id $downl $upl $del 0 $qs $max_steps ${experiment_id} ${constraints_id} ${threshold} ${max_actor_epochs} ${x1} ${x2} ${lambda_} ${original_model} ${snt_model_wo_ibp} ${k_symbolic_components} ${k} 1 &
                pids="$pids $!"
                act_id=$((act_id+1))
                act_port=$((port_base+act_id))
                sleep 2
            done
        done
    done

    for pid in $pids
    do
        echo "waiting for $pid"
        wait $pid
    done

    #Kill the learner
    echo "Killing the learner ..."
    sudo kill -s15 $lpid

    #Wait if it needs to save somthing!
    sleep 30

    #Make sure all are down ...
    for i in `seq 0 $((num_actors))`
    do   
        echo "Really killing the actors ..."
        sudo killall -s15 python
        sudo killall -s15 orca-server-mahimahi_v0
        sudo killall -s15 orca-server-mahimahi_v2
    done

    # Clean up the params.json
    sed "s/\"num_actors\"\: $num_actors/\"num_actors\"\: 1/" $cur_dir/params_base.json > "${dir}/params.json"
    # Make sure all are down ...
    sudo killall -s9 python client orca-server-mahimahi_v0 orca-server-mahimahi_v2
else
    echo "usage: $0 [{Learning from scratch=1} {Continue your learning=0} {Just Do Evaluation=4}] [base port number ]"
fi

sudo killall -s9 python client orca-server-mahimahi
