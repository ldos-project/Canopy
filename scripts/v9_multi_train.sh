# This experiments set the training reward as raw reward + symbolic constraint
# (1 - lambda_) * raw reward + lambda_ * symbolic reward
# Default Parameters
total_actors=32
port_base=44444
max_actor_epochs=50000
max_eval_actor_epochs=1100
original_model=0
snt_model_wo_ibp=0

x1=5

# Input parameters
if [ $# -ne 9 ]; then
    echo "Error in $0: Expecting 9 args, received $#"
    echo "./scripts/v9_run_multi.sh <lambda_> <k_symbolic_components> <k> <mode> <threshold> <seed> <constraints_id> <x2> <cloudlab_username>"
    exit 1
fi

lambda_=$1
k_symbolic_components=$2
k=$3
reward_mode=$4
threshold_unscaled=$5
seed=$6
constraints_id=$7 # The id of the constraint to use. 6 for safety; 7 for robustness.
x2=$8
cloudlab_username=$9

if [ "$threshold_unscaled" -le 0 ] || [ "$threshold_unscaled" -ge 100 ]; then
  echo "Argument <threshold> is out of range (0, 100). Exiting."
  echo "Usually this should be between 25 to 50"
  exit 1
fi

threshold=$(bc <<< "scale=2 ; $threshold_unscaled/100.0")


# Preprocess params.json
if [ ! -e ~/ConstrainedOrca/rl-module/params_distributed.json ]; then
    echo "[ERROR] params_distributed.json is missing on this node. Please recheck and try again"
    exit 1
fi

# All the v2 results were for empirical + raw reward.
num_actors=`python3 -c "import json; f = open('rl-module/params_distributed.json'); data = json.load(f)['actor_ip']; f.close(); print(len(data))"`
num_nodes=`python3 -c "import json; f = open('rl-module/params_distributed.json', 'r'); data = json.load(f)['actor_ip']; f.close(); num_nodes=len(set(map(lambda x: x.split(':')[0], data))); assert(num_nodes!=0); print(num_nodes)"`
num_actors_per_node=`echo "$num_actors/$num_nodes" | bc`

if [ -z "$num_actors" ] || [ -z "$num_nodes" ] || [ -z "$num_actors_per_node" ]; then
    echo "One of num_actors, num_nodes, num_actors_per_node: $num_actors, $num_nodes, $num_actors_per_node isn't set. Fatal error."
    exit 1
fi

if [ $((num_actors % num_nodes)) -ne 0 ]; then
    echo "Currently the script assumes that equal # of actors (usually, 32) on each node. Pls double check."
    exit 1
fi

experiment_name="v9_actorNum${num_actors}_multi_lambda${lambda_}_ksymbolic${k_symbolic_components}_k${k}_${reward_mode}_threshold${threshold_unscaled}_seed${seed}_constraints_id${constraints_id}_xtwo${x2}"

rm -rfv ~/intermediate_checkpoints/

# Run the training for only 1 session (?)
echo "About to start training session #1 at `date`"
./orca_v2_multi.sh 1 ${port_base} ${experiment_name} ${constraints_id} ${threshold} ${max_actor_epochs} ${x1} ${x2} ${lambda_} ${original_model} ${snt_model_wo_ibp} ${num_actors_per_node} ${k_symbolic_components} ${k} $reward_mode $seed ${cloudlab_username} 1
echo "Done with training session #1 at `date`"

# for train_session in 2 3 4 5; do
#     echo "Will save current train_dir to a new location before continuing training";
#     mkdir -p ~/intermediate_checkpoints/before_session_${train_session}/ 
#     cp -r ~/ConstrainedOrca/rl-module/train_dir/ ~/intermediate_checkpoints/before_session_${train_session}/

#     echo "About to start training session #${train_session} at `date`"
#     ./orca_v2_multi.sh 2 ${port_base} ${experiment_name} ${constraints_id} ${threshold} ${max_actor_epochs} ${x1} ${x2} ${lambda_} ${original_model} ${snt_model_wo_ibp} ${num_actors_per_node} ${k_symbolic_components} ${k} $reward_mode $seed ${cloudlab_username} ${train_session}
#     echo "Done with training session #${train_session} at `date`"
# done