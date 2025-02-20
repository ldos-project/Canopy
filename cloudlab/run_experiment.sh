#!/usr/bin/env bash
# 1: Name of the experiment
# 2: Start node of experiment
# 3: End node of experiment

NODE_PREFIX="node-"
EXP_NAME=$1
PROJECT_EXT=${CLOUDLAB_PROJECT}
DOMAIN=${CLOUDLAB_CLUSTER}
USER_NAME=${CLOUDLAB_USERNAME}
HOSTS=$(./cloudlab/nodes.sh $1 $2 $3)

TOTAL_NODES=$(($3 - $2 + 1))
echo "Starting experiment on ${TOTAL_NODES} nodes"

# Run command on every node except the control node
for host in ${HOSTS[@]}; do
  # Use awk to get the node number from the hostname.
  # Hostname format: node0.$DOMAIN.$PROJECT_EXT
  i=$(echo $host | awk -F'.' '{print $1}' | awk -F'node' '{print $2}')
  echo $host" "$i

  # Start in tmux session
  ssh -o StrictHostKeyChecking=no $host "tmux new-session -d -s orca \"cd \$HOME; ./scripts/v9_run_multi.sh 1 10 1 sym $i ${TOTAL_NODES}; bash\""
done