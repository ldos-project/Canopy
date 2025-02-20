if [ "$#" -ne 11 ]; then
  echo "run_eval.sh [ERROR] Only $# args found. Expected 10"
  exit 1
fi

model_name=$1 # model_name must include raw-sym, etc.
trace_path=$2
eval_k_steps=$3 # default 1
eval_k_symbolic_components=$4 # default 50
reward_mode=$5 # default raw-sym
max_eval_actor_epochs=$6
seed=$7
constraints_id=$8 # constraints_id=6
eval_threshold=$9 # threshold=0.25
x2=${10} # x2=15
bdp_multiplier=${11}
# eval_threshold=0.25

# Default Parameters
num_actors=1
port_base=44444
original_model=0
snt_model_wo_ibp=0

x1=5

lambda_=0.5 # not used.

# Run evaluation.
./eval.sh 4 \
    ${port_base} \
    ${model_name} \
    ${constraints_id} \
    ${eval_threshold} \
    ${max_eval_actor_epochs} \
    ${x1} \
    ${x2} \
    ${lambda_} \
    ${original_model} \
    ${snt_model_wo_ibp} \
    ${num_actors} \
    ${eval_k_symbolic_components} \
    ${eval_k_steps} \
    ${trace_path} \
    ${reward_mode} \
    ${seed} \
    ${bdp_multiplier}