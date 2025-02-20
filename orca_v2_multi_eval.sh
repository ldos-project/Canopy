if [ $# -eq 14 ]
then
    source setup.sh

    first_time=$1
    port_base=$2
    experiment_id=$3
    constraints_id=$4
    threshold=$5
    max_actor_epochs=$6
    x1=$7
    x2=$8
    lambda_=$9
    original_model=${10}
    snt_model_wo_ibp=${11}
    cur_dir=`pwd -P`
    scheme_="cubic"
    max_steps=500000        #Run until you collect 50k samples per actor
    eval_duration=30000000
    num_actors=${12}
    k_symbolic_components=${13}
    k=${14}
    memory_size=$((max_steps*30))
    dir="${cur_dir}/rl-module"

    if [ $num_actors -eq 1 ]
    then
        bw_array=(48)
    elif [ $num_actors -eq 32 ]
    then
        bw_array=($(seq 6 6 192))
    else
        echo "Please specify num_actors."
    fi

    sed "s/\"num_actors\"\: 1/\"num_actors\"\: 1/" $cur_dir/params_base.json > "${dir}/params.json"
    sed -i "s/\"memsize\"\: 2553600/\"memsize\"\: $memory_size/" "${dir}/params.json"
    sudo killall -s9 python client orca-server-mahimahi_v0 orca-server-mahimahi_v2

    epoch=20
    act_port=$port_base
    tmp_num_actor=32
    if [ $1 -eq 4 ]
    then
       # If you are here: You are going to perform an evaluation over an emulated link
       sed "s/\"num_actors\"\: 1/\"num_actors\"\: $tmp_num_actor/" $cur_dir/params_base.json > "${dir}/params.json"

       echo "./learner_v2.sh  $dir $first_time $experiment_id $constraints_id $threshold $max_actor_epochs $x1 $x2 $lambda_ $original_model $snt_model_wo_ibp &"
       ./learner_v2.sh  $dir ${first_time} ${experiment_id} ${constraints_id} ${threshold} ${max_actor_epochs} ${x1} ${x2} ${lambda_} ${original_model} ${snt_model_wo_ibp} ${k_symbolic_components} ${k} &
       #Bring up the actors:
       act_id=0
       # Only evaluate this.
        # downl="step-10s-3-level"
        for dl in "${bw_array[@]}"
        do
            # dl=48
            downl="wired$dl"
            echo "downl=$downl"
            upl="wired48"
            for del in 10
            do
                bdp=$((2*dl*del/12))     #12Mbps=1pkt per 1 ms ==> BDP=2*del*BW=2*del*dl/12
                for qs in $((2*bdp))
                do
                    echo "Start actor with actor_id=$act_id"
                    ./actor_v2.sh ${act_port} $epoch ${first_time} $scheme_ $dir $act_id $downl $upl $del $eval_duration $qs $max_steps ${experiment_id} ${constraints_id} ${threshold} ${max_actor_epochs} ${x1} ${x2} ${lambda_} ${original_model} ${snt_model_wo_ibp} ${k_symbolic_components} ${k} 0 &
                    pids="$pids $!"
                    act_id=$((act_id+1))
                    act_port=$((port_base+act_id))
                    sleep 2
                done
            done
            if [ $act_id -eq $tmp_num_actor ]
            then
                break
            fi
        done
        for pid in $pids
        do
            echo "waiting for $pid"
            wait $pid
        done
        #Bring down the learner and actors ...
        for i in `seq 0 $((num_actors))`
        do
            sudo killall -s15 python
            sudo killall -s15 orca-server-mahimahi_v0
            sudo killall -s15 orca-server-mahimahi_v2
            sudo killall -s15 client
        done
    else
      echo "Only do evaluation [first_time=4]"
    fi
    # Make sure all are down ...
    sudo killall -s9 python client orca-server-mahimahi_v0 orca-server-mahimahi_v2

    sed "s/\"num_actors\"\: $tmp_num_actor/\"num_actors\"\: 1/" $cur_dir/params_base.json > "${dir}/params.json"
else
    echo "usage: $0 [{Learning from scratch=1} {Continue your learning=0} {Just Do Evaluation=4}] [base port number ]"
fi

sudo killall -s9 python client orca-server-mahimahi
