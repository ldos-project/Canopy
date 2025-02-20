#!/usr/bin/env bash
# 1: Name of the experiment
# 2: Start node of experiment
# 3: End node of experiment
# 4: cluster
# 5: cloudlab username

declare -A clusternames
clusternames[emu]="emulab.net"
clusternames[wisc]="wisc.cloudlab.us"
clusternames[utah]="utah.cloudlab.us"

if [[ ! -v clusternames[$4] ]]; then
  echo "'$4' is not a valid cluster."
  # You can exit or continue as needed
  exit 0
fi

NODE_PREFIX="node-"
HOSTS=`./cloudlab/nodes.sh $1 $2 $3 ${clusternames[$4]} $5 --all | tr -d ' ' | xargs`

# Run command on every node except the control node
for host in $HOSTS; do
  echo $host
  # ssh -o StrictHostKeyChecking=no $host "uname -r"
  # ssh -o StrictHostKeyChecking=no $host "source ~/venv/bin/activate && pip install git+https://github.com/deepmind/interval-bound-propagation"
  
  # Copy over the training checkpoints
  ssh -o StrictHostKeyChecking=no $host "mkdir -p ~/ConstrainedOrca/rl-module/train_dir; cp -r /proj/verifiedmlsys-PG0/rohitd99/model_checkpoints_256_actors/sensitivity_ksc_1/seed0/learner0-v9_actorNum256_multi_lambda0.25_ksymbolic1_k1_raw-sym_threshold25_seed0 ~/ConstrainedOrca/rl-module/train_dir/c3-learner0-session1-v9_actorNum256_multi_lambda0.25_ksymbolic1_k1_raw-sym_threshold25_seed0"

  # In a file checkpoint, replace "rohitd99" with "dsaxena" and "seed0/learner0" with "c3-learner0-session1"
  ssh -o StrictHostKeyChecking=no $host "sudo sed -i 's/rohitd99/dsaxena/g' ~/ConstrainedOrca/rl-module/train_dir/c3-learner0-session1-v9_actorNum256_multi_lambda0.25_ksymbolic1_k1_raw-sym_threshold25_seed0/checkpoint"
  ssh -o StrictHostKeyChecking=no $host "sudo sed -i 's/seed0\/learner0/c3-learner0-session1/g' ~/ConstrainedOrca/rl-module/train_dir/c3-learner0-session1-v9_actorNum256_multi_lambda0.25_ksymbolic1_k1_raw-sym_threshold25_seed0/checkpoint"

  # Chmod /mydata/log
  # ssh -o StrictHostKeyChecking=no $host "sudo chmod -R 777 /mydata/log"
done
