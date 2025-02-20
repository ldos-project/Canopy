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
    dlbw=${12}
    k_symbolic_components=${13}
    k=${14}
    cur_dir=`pwd -P`
    scheme_="cubic"
    max_steps=500000         #Run untill you collect 50k samples per actor
    eval_duration=30
    num_actors=1
    memory_size=$((max_steps*num_actors))
    dir="${cur_dir}/rl-module"

    sed "s/\"num_actors\"\: 1/\"num_actors\"\: $num_actors/" $cur_dir/params_base.json > "${dir}/params.json"
    sed -i "s/\"memsize\"\: 5320000/\"memsize\"\: $memory_size/" "${dir}/params.json"
    sudo killall -s9 python client orca-server-mahimahi_v0 orca-server-mahimahi_v2

    epoch=20
    act_port=$port_base

    if [ $1 -eq 4 ]
    then
       # If you are here: You are going to perform an evaluation over an emulated link
       num_actors=1
       sed "s/\"num_actors\"\: 1/\"num_actors\"\: $num_actors/" $cur_dir/params_base.json > "${dir}/params.json"

       echo "./learner_v2.sh  $dir $first_time $experiment_id $constraints_id $threshold $max_actor_epochs $x1 $x2 $lambda_ $original_model $snt_model_wo_ibp $k_symbolic_components $k &"
       ./learner_v2.sh  $dir ${first_time} ${experiment_id} ${constraints_id} ${threshold} ${max_actor_epochs} ${x1} ${x2} ${lambda_} ${original_model} ${snt_model_wo_ibp} ${k_symbolic_components} ${k} &
       #Bring up the actors:
       act_id=0
       # for dl in 48
       for dl in ${dlbw}
       do
           downl="wired$dl"
           echo "downl=$downl"
           upl="wired48"
           for del in 10
           do
               bdp=$((2*dl*del/12))     #12Mbps=1pkt per 1 ms ==> BDP=2*del*BW=2*del*dl/12
               for qs in $((2*bdp))
               do
                   ./actor_v2.sh ${act_port} $epoch ${first_time} $scheme_ $dir $act_id $downl $upl $del $eval_duration $qs 0 ${experiment_id} ${constraints_id} ${threshold} ${max_actor_epochs} ${x1} ${x2} ${lambda_} ${original_model} ${snt_model_wo_ibp} ${k_symbolic_components} ${k} 0 &
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
        #Bring down the learner and actors ...
        for i in `seq 0 $((num_actors))`
        do
            sudo killall -s15 python
            sudo killall -s15 orca-server-mahimahi_v0
            sudo killall -s15 orca-server-mahimahi_v2
            sudo killall -s15 client
        done
    else
    # If you are here: You are going to start/continue learning a better model!

      #Bring up the learner:
      echo "./learner_v2.sh  $dir $first_time $experiment_id ${constraints_id} ${threshold} &"
      if [ $1 -eq 1 ];
      then
          # Start the learning from the scratch
           /users/`logname`/venv/bin/python ${dir}/d5_v2.py --job_name=learner --task=0 --base_path=${dir} --experiment_id=${experiment_id} --constraints_id=${constraints_id} --threshold=${threshold} --max_actor_epochs=${max_actor_epochs} --x1=${x1} --x2=${x2} --lambda_=${lambda_} --original_model=${original_model} --snt_model_wo_ibp=${snt_model_wo_ibp} --k_symbolic_components=${k_symbolic_components} --k=${k} --only_tcp=0 &
           lpid=$!
       else
          # Continue the learning on top of previous model
           /users/`logname`/venv/bin/python ${dir}/d5_v2.py --job_name=learner --task=0 --base_path=${dir} --load --experiment_id=${experiment_id} --constraints_id=${constraints_id} --threshold=${threshold} --max_actor_epochs=${max_actor_epochs} --x1=${x1} --x2=${x2} --lambda_=${lambda_} --original_model=${original_model} --snt_model_wo_ibp=${snt_model_wo_ibp} --k_symbolic_components=${k_symbolic_components} --k=${k} --only_tcp=0 &
           lpid=$!
       fi
       sleep 10

       #Bring up the actors:
       # Here, we go with single actor
       act_id=0
       # for dl in 48
       for dl in ${dlbw}
       do
           downl="wired$dl"
           echo "downl=$downl"
           upl=$downl
           for del in 10
           do
               bdp=$((2*dl*del/12))      #12Mbps=1pkt per 1 ms ==> BDP=2*del*BW=2*del*dl/12
               for qs in $((2*bdp))
               do
                   echo "Start actor $act_id"
                   ./actor_v2.sh ${act_port} $epoch ${first_time} $scheme_ $dir $act_id $downl $upl $del 0 $qs $max_steps ${experiment_id} ${constraints_id} ${threshold} ${max_actor_epochs} ${x1} ${x2} ${lambda_} ${original_model} ${snt_model_wo_ibp} ${k_symbolic_components} ${k} 0 &
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
    fi
    # Make sure all are down ...
    sudo killall -s9 python client orca-server-mahimahi_v0 orca-server-mahimahi_v2
else
    echo "usage: $0 [{Learning from scratch=1} {Continue your learning=0} {Just Do Evaluation=4}] [base port number ]"
fi

sudo killall -s9 python client orca-server-mahimahi

