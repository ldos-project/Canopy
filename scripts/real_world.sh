if [ "$#" -ne 6 ]; then
  echo "run_eval.sh [ERROR] Only $# args found. Expected 6"
  echo "args: <model_name> <link_name> <constraints_id> <eval_threshold> <x2> <run>"
  exit 1
fi

sudo killall -s9 python client orca-server-mahimahi_v2

model_name=$1 # model_name must include raw-sym, etc.
link_name=$2
eval_k_steps=1
eval_k_symbolic_components=25
reward_mode="raw-sym"
max_eval_actor_epochs=1
seed=0
constraints_id=$3
eval_threshold=$4 
x2=${5}
run=$6

port_base=44444
cur_dir=`pwd -P`
dir="${cur_dir}/rl-module"

learned_model() {
    sudo sysctl -w net.ipv4.tcp_congestion_control=cubic
    # Default Parameters
    num_actors=1
    
    original_model=0
    snt_model_wo_ibp=0
    x1=5
    lambda_=0.5
    cur_dir=`pwd -P`
    first_time=4

    if [ $model_name == "tcp-cubic" ]
    then
        only_tcp=1
    else
        only_tcp=0
    fi

    ./learner_v2.sh  $dir \
            ${first_time} \
            ${model_name} \
            ${constraints_id} \
            ${eval_threshold} \
            ${max_eval_actor_epochs} \
            ${x1} \
            ${x2} \
            ${lambda_} \
            ${original_model} \
            ${snt_model_wo_ibp} \
            ${eval_k_symbolic_components} \
            ${eval_k_steps} \
            ${only_tcp} \
            ${link_name} \
            ${reward_mode} \
            ${seed} &

    act_id=0
    act_port=44444
    period=20
    scheme="cubic"
    eval_duration=60000000
    max_steps=500000

    echo "Start actor with actor_id=$act_id"

    log="orca-$scheme-${period}-${model_name}-${link_name}-${constraints_id}-${eval_threshold}-${seed}-run${run}"
    training_session_idx=-1

    $dir/real_world \
        $act_port \
        $dir \
        ${period} \
        ${first_time} \
        $scheme \
        $act_id \
        $log \
        $eval_duration \
        $max_steps \
        ${model_name} \
        ${constraints_id} \
        ${eval_threshold} \
        ${max_eval_actor_epochs} \
        ${x1} \
        ${x2} \
        ${lambda_} \
        ${original_model} \
        ${snt_model_wo_ibp} \
        ${eval_k_symbolic_components} \
        ${eval_k_steps} \
        ${only_tcp} \
        ${link_name} \
        ${reward_mode} \
        ${seed} \
        ${training_session_idx} 
}

tcp () {
    sudo sysctl -w net.ipv4.tcp_congestion_control=$model_name
    log="tcp-${model_name}-${link_name}-run${run}"
    ${dir}/tcp_eval ${port_base} ${dir} ${model_name} $link_name none 0 ${log} 60 0
}

case "$model_name" in
    cubic|bbr|reno|vegas)
        echo $model_name is a TCP variant.
        tcp
        ;;
    *)
        echo $model_name is a learned model.
        learned_model
        ;;
esac
