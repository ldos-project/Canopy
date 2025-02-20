# Arguments:
# $1: model_name
# $2: trace directory
# $3: results directory
model_name=$1
trace_dir=$2
results_dir=$3

# Check if number of arguments is correct.
if [ "$#" -ne 3 ]
then
    echo "Usage: $0 <model_name> <trace_dir> <results_dir>"
    exit 1
fi

fixed_bw_traces=("wired6" "wired12" "wired24" "wired48" "wired96" "wired192")

pushd $HOME/ConstrainedOrca

# Run for 3 runs.
for run in 0 1 2
do
    # Create a directory for the run.
    mkdir -p ${results_dir}/run${run}

    # Iterate over all traces in the trace directory.
    for trace in $(ls $trace_dir)
    do
        # Check if the trace is a file.
        if [ -f $trace_dir/$trace ]
        then
            # If the trace is a fixed bandwidth trace, then run only if it is in the fixed_bw_traces list.
            if [[ $trace == *"wired"* ]]
            then
                if [[ ! " ${fixed_bw_traces[@]} " =~ " ${trace} " ]]
                then
                    echo "Skipping $trace. Fixed bandwidth trace not in traces of concern."
                    continue
                fi

                continue
            fi

            echo "[INFO] Running robustness evaluation for $trace."

            mkdir -p ${results_dir}/run${run}/${trace}-eval

            # Evaluate baseline model.
            ./robustness.sh $model_name 0 0 0.05 uniform $trace
            # Move the evaluation log files to the results directory.
            mkdir -p ${results_dir}/run${run}/${trace}-eval/baseline
            cp $HOME/ConstrainedOrca/rl-module/evaluation_log/seed0/*${trace}* ${results_dir}/run${run}/${trace}-eval/baseline/

            # Evaluate model with noise.
            # ./robustness.sh $model_name 1 0 0.05 uniform $trace
            # mkdir -p ${results_dir}/run${run}/${trace}-eval/noise-Thr-0.05
            # cp $HOME/ConstrainedOrca/rl-module/evaluation_log/seed0/*${trace}* ${results_dir}/run${run}/${trace}-eval/noise-Thr-0.05/
            # sleep 5

            ./robustness.sh $model_name 1 5 0.05 uniform $trace
            mkdir -p ${results_dir}/run${run}/${trace}-eval/noise-Delay-0.05
            cp $HOME/ConstrainedOrca/rl-module/evaluation_log/seed0/*${trace}* ${results_dir}/run${run}/${trace}-eval/noise-Delay-0.05/
            sleep 5

            # ./robustness.sh $model_name 1 0 0.1 uniform $trace
            # mkdir -p ${results_dir}/run${run}/${trace}-eval/noise-Thr-0.1
            # cp $HOME/ConstrainedOrca/rl-module/evaluation_log/seed0/*${trace}* ${results_dir}/run${run}/${trace}-eval/noise-Thr-0.1/
            # sleep 5

            # ./robustness.sh $model_name 1 5 0.1 uniform $trace
            # mkdir -p ${results_dir}/run${run}/${trace}-eval/noise-Delay-0.1
            # cp $HOME/ConstrainedOrca/rl-module/evaluation_log/seed0/*${trace}* ${results_dir}/run${run}/${trace}-eval/noise-Delay-0.1/

            # Move the result files from /mydata/log to the results directory.
            mv /mydata/log/$trace ${results_dir}/run${run}/

            echo "[INFO] Evaluation completed for $trace. Sleeping 5 seconds."
            sleep 5
        else
            echo "Skipping $trace. Not a file."
        fi
    done
done

popd