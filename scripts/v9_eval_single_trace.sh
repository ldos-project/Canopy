# This experiments set the training reward as raw reward + symbolic constraint
# (1 - lambda_) * raw reward + lambda_ * symbolic reward

# Default Parameters
num_actors=32
port_base=44444
constraints_id=6
max_actor_epochs=1100
original_model=0
snt_model_wo_ibp=0
threshold=0.25
x1=5
x2=15

# Which model is evaluated on?
# Input parameters
lambda_=$1
k_symbolic_components=$2
k=$3
mode=$4

# Customize the evaluation configs.
eval_k_array=(1 3 1 3 1 3)
eval_k_symbolic_components_array=(25 25 50 50 100 100)
eval_threshold_array=(25 25 25 25 25 25)

for idx in {0,1,2,3,4,5}; do
    eval_k=${eval_k_array[$idx]}
    eval_k_symbolic_components=${eval_k_symbolic_components_array[$idx]}
    eval_threshold=${eval_threshold_array[$idx]}

    # All the v2 results were for empirical + raw reward.
    experiment_id="v9_actorNum"${num_actors}"_multi_"${mode}"."${eval_k}"."${eval_k_symbolic_components}"."${eval_threshold}
    
    #TODO: 
    # - Automatically generate a mapping from parameters to experiment_name.
    # - only use expr name in eval with learner0
    # - check if all the actors's behaviors are the same as learner0
    # Evaluation
    ./orca_v2_multi.sh 4 ${port_base} ${experiment_id} ${constraints_id} ${threshold} ${max_actor_epochs} ${x1} ${x2} ${lambda_} ${original_model} ${snt_model_wo_ibp} ${num_actors} ${k_symbolic_components} ${k}

done