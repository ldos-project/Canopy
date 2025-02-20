# This experiments set the training reward as raw reward + symbolic constraint
# (1 - lambda_) * raw reward + lambda_ * symbolic reward
# Default Parameters
num_actors=32 # FIXME: Change this to 32
port_base=44444
constraints_id=6
max_actor_epochs=100000
max_eval_actor_epochs=1100
original_model=0
snt_model_wo_ibp=0
threshold=0.25
x1=5
x2=15

# Input parameters
lambda_=$1
k_symbolic_components=$2
k=$3

# All the v2 results were for empirical + raw reward.
experiment_id="v9_actorNum"${num_actors}"_multi_only-heu"

# Run the training for 5 sessions.
./orca_v2_multi.sh 1 ${port_base} ${experiment_id} ${constraints_id} ${threshold} ${max_actor_epochs} ${x1} ${x2} ${lambda_} ${original_model} ${snt_model_wo_ibp} ${num_actors} ${k_symbolic_components} ${k}
echo "First time done"
./orca_v2_multi.sh 2 ${port_base} ${experiment_id} ${constraints_id} ${threshold} ${max_actor_epochs} ${x1} ${x2} ${lambda_} ${original_model} ${snt_model_wo_ibp} ${num_actors} ${k_symbolic_components} ${k}
echo "Second time done"
./orca_v2_multi.sh 2 ${port_base} ${experiment_id} ${constraints_id} ${threshold} ${max_actor_epochs} ${x1} ${x2} ${lambda_} ${original_model} ${snt_model_wo_ibp} ${num_actors} ${k_symbolic_components} ${k}
echo "Third time done"
./orca_v2_multi.sh 2 ${port_base} ${experiment_id} ${constraints_id} ${threshold} ${max_actor_epochs} ${x1} ${x2} ${lambda_} ${original_model} ${snt_model_wo_ibp} ${num_actors} ${k_symbolic_components} ${k}
echo "Fourth time done"
./orca_v2_multi.sh 2 ${port_base} ${experiment_id} ${constraints_id} ${threshold} ${max_actor_epochs} ${x1} ${x2} ${lambda_} ${original_model} ${snt_model_wo_ibp} ${num_actors} ${k_symbolic_components} ${k}
echo "Fifth time done"

# Evaluation
# The problem is here. The evaluation never ends..., why?
./orca_v2_multi.sh 4 ${port_base} ${experiment_id} ${constraints_id} ${threshold} ${max_eval_actor_epochs} ${x1} ${x2} ${lambda_} ${original_model} ${snt_model_wo_ibp} ${num_actors} ${k_symbolic_components} ${k}
