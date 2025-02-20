if [ $# -eq 17 ] || [ $# -eq 18 ]; then
    source setup.sh

    first_time=$1
    port_base=$2
    model_name=$3
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
    max_steps=55000        # MAX_SAMPLES_PER_ACTOR
    eval_duration=30000000
    num_actors=${12}        #Number of actors on each node
    k_symbolic_components=${13}
    k=${14}
    trace_name="TRAIN_TRACE"
    reward_mode=${15}
    seed=${16}
    cloudlab_username=${17}
    if [ "$constraints_id" -eq 11 ]; then bdp_multiplier=5
    elif [ "$constraints_id" -eq 12 ]; then bdp_multiplier=0.5
    else bdp_multiplier=2
    fi

    if [ $# -eq 18 ]; then
        training_session_idx=${18}
    else
        echo "[orca_v2_multi.sh] training_session_idx not set"
        training_session_idx=-1
    fi

    ACTOR_NODE_IPS=`python3 -c "import json; f = open('rl-module/params_distributed.json'); data = json.load(f)['actor_ip']; f.close(); node_ips=set(map(lambda x: x.split(':')[0], data)); print(' '.join(node_ips))"`
    LEARNER_NODE_IP=`python3 -c "import json; f = open('rl-module/params_distributed.json'); learner_ip = json.load(f)['learner_ip']; f.close(); print(learner_ip.split(':')[0])"`

    echo "Make sure learner node and all actor nodes have no running processes before beginning..."
    echo "Found actor IPs: $ACTOR_NODE_IPS"
    sudo killall -s9 python client orca-server-mahimahi_v0 orca-server-mahimahi_v2 || true
    for actor_ip in $ACTOR_NODE_IPS
    do  
        ssh -o StrictHostKeyChecking=no ${cloudlab_username}@$actor_ip "sudo killall -s15 python client orca-server-mahimahi_v0 orca-server-mahimahi_v2; ps aux | grep "[C]onstrainedOrca" | tr -s ' ' | cut -d ' ' -f2 | xargs kill -9;" || true
    done

    bw_array=`echo $(seq 6 6 192) | xargs`
    del_array="10 2 25 50 75 100 150 200"

    memory_size=$((max_steps*$num_actors))
    dir="${cur_dir}/rl-module"

    epoch=20
    
    # Setup params.json on all nodes.
    
    sed "s/\"memsize\"\: 2553600/\"memsize\"\: $memory_size/" ~/ConstrainedOrca/rl-module/params_distributed.json > ~/ConstrainedOrca/rl-module/params.json

    for actor_ip in $ACTOR_NODE_IPS; do
        echo "Pushing params.json to $actor_ip"
        scp -o StrictHostKeyChecking=no ~/ConstrainedOrca/rl-module/params.json $cloudlab_username@$actor_ip:~/ConstrainedOrca/rl-module/params.json
    done

    #Bring up the learner:
    if [ $1 -eq 4 ];
    then
       echo "[WARN] [WARN] [WARN] You are trying to use an unspoorted codepath!!!"
       echo "This codepath was NOT fixed when master was merged into distributed."
       echo "Fatal error. Exiting"
       exit 1
       
       # If you are here: You are going to perform an evaluation over an emulated link
       echo "./learner_v2.sh  $dir $first_time $model_name $constraints_id $threshold $max_actor_epochs $x1 $x2 $lambda_ $original_model $snt_model_wo_ibp &"
       ./learner_v2.sh  $dir ${first_time} ${model_name} ${constraints_id} ${threshold} ${max_actor_epochs} ${x1} ${x2} ${lambda_} ${original_model} ${snt_model_wo_ibp} ${k_symbolic_components} ${k} &
       #Bring up the actors:
       act_id=0
       ALL_ACTOR_IP_ADDR=""
       echo "Set BDP multiplier: $bdp_multiplier"
       for dl in $bw_array
       do
           downl="wired$dl"
           echo "downl=$downl"
           upl="wired48"
           for del in $del_array
           do
               bdp=$((2*dl*del/12))     #12Mbps=1pkt per 1 ms ==> BDP=2*del*BW=2*del*dl/12               
               qs_arr=$(echo "$bdp_multiplier * $bdp" | bc -l | awk '{print int($1)}')
               for qs in $qs_arr
               do
                   echo "[EVAL] Starting actor with actor_id=$act_id"
                   actor_ip_addr=`python3 -c "import json; f = open('rl-module/params.json'); data = json.load(f)['actor_ip']; f.close(); print(data[$act_id].split(':')[0])"`
                   actor_port=`python3 -c "import json; f = open('rl-module/params.json'); data = json.load(f)['actor_ip']; f.close(); print(data[$act_id].split(':')[1])"`  
                   
                   if [ -z "$actor_ip_addr" ]; then
                      echo "It looks like $act_id does not exist in params.json"
                      echo "Assuming that $((act_id-1)) idx in params.json is the last actor"
                      break 3
                   fi

                   mahimahi_port=$((port_base+act_id))                   
                   
                   this_actor_pid=`ssh -o StrictHostKeyChecking=no $cloudlab_username@$actor_ip_addr "nohup ./ConstrainedOrca/actor_v2.sh ${mahimahi_port} $epoch ${first_time} $scheme_ $dir $act_id $downl $upl $del $eval_duration $qs $max_steps ${model_name} ${constraints_id} ${threshold} ${max_actor_epochs} ${x1} ${x2} ${lambda_} ${original_model} ${snt_model_wo_ibp} ${k_symbolic_components} ${k} 0 $training_session_idx >> ~/actor_logs/actor_${act_id} 2>&1 & echo \\$!"`
                   echo "[EVAL] Started actor $act_id on $actor_ip_addr:$actor_port (MM port: $mahimahi_port) with PID $this_actor_pid [delay=$del; bw=$dl; bdp=$bdp; qs=$qs]"

                   if [ -z "$this_actor_pid" ]; then
                        echo "this_actor_pid was not set in this iteration. Fatal error"
                   else
                        ALL_ACTOR_IP_ADDR="$ALL_ACTOR_IP_ADDR $actor_ip_addr:$this_actor_pid"
                   fi

                   act_id=$((act_id+1))
                   sleep 0.5
               done
           done
       done

       act_id=0
       for ip_pid in $ALL_ACTOR_IP_ADDR
       do
           this_actor_ip=`echo $ip_pid | cut -d ':' -f1`
           this_actor_pid=`echo $ip_pid | cut -d ':' -f2`
           echo "Waiting for actor#$act_id with pid $this_actor_pid on node $this_actor_ip"
           ssh -o StrictHostKeyChecking=no $cloudlab_username@$this_actor_ip "while kill -0 $this_actor_pid 2>/dev/null; do sleep 1; done"
           act_id=$((act_id+1))
       done

       echo "Done waiting for all actor PIDs. Now will double check to see if any d5_v2.py instances are running: $ACTOR_NODE_IPS"
       for this_actor_ip in $ACTOR_NODE_IPS; do
        if [ "$LEARNER_NODE_IP" != "$this_actor_ip" ]; then
            echo "[START double check] $this_actor_ip"
            ssh -o StrictHostKeyChecking=no $cloudlab_username@$this_actor_ip "while [ \$(ps aux | grep '[d]5_v2.py' | wc -l) -ne 0 ]; do sleep 1; done"
            echo "[END double check] $this_actor_ip"
        fi
       done

        # Bring down the learner and actors ...
        for actor_ip in $ACTOR_NODE_IPS
        do
            ssh -o StrictHostKeyChecking=no $cloudlab_username@$actor_ip "sudo killall -s15 python; sudo killall -s15 orca-server-mahimahi_v2 orca-server-mahimahi"
        done
    else
    # If you are here: You are going to start/continue learning a better model!

      #Bring up the learner:
      if [ $1 -eq 1 ];
      then
          for c_actor_ip in $ACTOR_NODE_IPS 
          do 
            echo "Cleanup ${c_actor_ip}'s actor_logs since we are coming up the first time"
            ssh -o StrictHostKeyChecking=no ${cloudlab_username}@$c_actor_ip rm -rfv ~/actor_logs/* ~/train_dir/
          done
          # Start the learning from the scratch
           /users/`logname`/venv/bin/python ${dir}/d5_v2.py --job_name=learner --task=0 --base_path=${dir} --model_name=${model_name} --constraints_id=${constraints_id} --threshold=${threshold} --max_actor_epochs=${max_actor_epochs} --x1=${x1} --x2=${x2} --lambda_=${lambda_} --original_model=${original_model} --snt_model_wo_ibp=${snt_model_wo_ibp} --k_symbolic_components=${k_symbolic_components} --k=${k} --only_tcp=0 --trace_name=${trace_name} --reward_mode=${reward_mode} --seed=${seed} --training_session_idx=${training_session_idx} &
           lpid=$!
       else
          # Continue the learning on top of previous model
           /users/`logname`/venv/bin/python ${dir}/d5_v2.py --job_name=learner --task=0 --base_path=${dir} --load --model_name=${model_name} --constraints_id=${constraints_id} --threshold=${threshold} --max_actor_epochs=${max_actor_epochs} --x1=${x1} --x2=${x2} --lambda_=${lambda_} --original_model=${original_model} --snt_model_wo_ibp=${snt_model_wo_ibp} --k_symbolic_components=${k_symbolic_components} --k=${k} --only_tcp=0 --trace_name=${trace_name} --reward_mode=${reward_mode} --seed=${seed} --training_session_idx=${training_session_idx} &
           lpid=$!
       fi
       sleep 10

       act_id=0
       ALL_ACTOR_IP_ADDR=""


       for del in $del_array # minRTT is 2*del
       do
           for dl in $bw_array
           do
               downl="wired$dl"
               echo "downl=$downl"
               upl="wired48"
               bdp=$((2*dl*del/12))      #12Mbps=1pkt per 1 ms ==> BDP=2*del*BW=2*del*dl/12
               qs_arr=$(echo "$bdp_multiplier * $bdp" | bc -l | awk '{print int($1)}')

               for qs in $qs_arr
               do
                   actor_ip_addr=`python3 -c "import json; f = open('rl-module/params_distributed.json'); data = json.load(f)['actor_ip']; f.close(); print(data[$act_id].split(':')[0])"` || true 2> /dev/null
                   actor_port=`python3 -c "import json; f = open('rl-module/params_distributed.json'); data = json.load(f)['actor_ip']; f.close(); print(data[$act_id].split(':')[1])"` || true 2> /dev/null

                   if [ -z "$actor_ip_addr" ]; then
                      echo "It looks like $act_id does not exist in params.json"
                      echo "Assuming that $((act_id-1)) idx in params.json is the last actor"
                      break 3
                   fi

                   mahimahi_port=$((port_base+act_id))
                   this_actor_pid=`ssh -o StrictHostKeyChecking=no $cloudlab_username@$actor_ip_addr "nohup ./ConstrainedOrca/actor_v2.sh ${mahimahi_port} $epoch ${first_time} $scheme_ $dir $act_id $downl $upl $del 0 $qs $max_steps ${model_name} ${constraints_id} ${threshold} ${max_actor_epochs} ${x1} ${x2} ${lambda_} ${original_model} ${snt_model_wo_ibp} ${k_symbolic_components} ${k} 0 $trace_name $reward_mode $seed $training_session_idx >> ~/actor_logs/actor_${act_id} 2>&1 & echo \\$!"`

                   if [ -z "$this_actor_pid" ]; then
                        echo "this_actor_pid was not set in this iteration. Fatal error"
                        exit 1
                   fi

                   echo "[TRAIN] Started actor $act_id on $actor_ip_addr:$actor_port (MM port: $mahimahi_port) with PID $this_actor_pid [delay=$del; bw=$dl; bdp=$bdp; qs=$qs]"

                   ALL_ACTOR_IP_ADDR="$ALL_ACTOR_IP_ADDR $actor_ip_addr:$this_actor_pid"
                   act_id=$((act_id+1))
                   sleep 0.5
               done
           done
       done
       
       act_id=0
       for ip_pid in $ALL_ACTOR_IP_ADDR
       do
           this_actor_ip=`echo $ip_pid | cut -d ':' -f1`
           this_actor_pid=`echo $ip_pid | cut -d ':' -f2`
           echo "[TRAIN] Waiting for actor #$act_id with pid $this_actor_pid on node $this_actor_ip"
           if [ "$act_id" -eq 0 ]; then
              echo "No timeout since act_id=0"
              ssh -o StrictHostKeyChecking=no $cloudlab_username@$this_actor_ip "while kill -0 $this_actor_pid 2>/dev/null; do sleep 1; done"
           else
              echo "100s timeout since act_id != 0"
              timeout 100 ssh -o StrictHostKeyChecking=no $cloudlab_username@$this_actor_ip "while kill -0 $this_actor_pid 2>/dev/null; do sleep 1; done"
           fi
           
           act_id=$((act_id+1))
       done

       echo "[TRAIN] Done waiting for all actor PIDs. Now will double check to see if any d5_v2.py instances are running"
       for this_actor_ip in $ACTOR_NODE_IPS; do
        if [ "$LEARNER_NODE_IP" != "$this_actor_ip" ]; then
            echo "[TRAIN] START double check for $this_actor_ip"
            ssh -o StrictHostKeyChecking=no $cloudlab_username@$this_actor_ip "while [ \$(ps aux | grep '[d]5_v2.py' | wc -l) -ne 0 ]; do sleep 1; done"
            echo "[TRAIN] END double check for $this_actor_ip"
        fi
       done
       
       # Kill the learner
       echo "Killing the learner ..."
       sudo kill -s15 $lpid

       # Wait if it needs to save somthing!
       sleep 30

       # Make sure all are down ...
       echo "Really killing the actors ..."
       for actor_ip in $ACTOR_NODE_IPS
       do
           ssh -o StrictHostKeyChecking=no $cloudlab_username@$actor_ip "sudo killall -s9 python; sudo killall -s9 orca-server-mahimahi_v2 orca-server-mahimahi;"
       done
    fi
else
    echo "usage: $0 [{Learning from scratch=1} {Continue your learning=0} {Just Do Evaluation=4}] [base port number ]"
fi

# Make sure all are down ...
sudo killall -s9 python client orca-server-mahimahi orca-server-mahimahi_v2
