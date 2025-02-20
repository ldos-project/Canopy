# Arguments:
# $1: model_name
# $2: trace directory
# $3: results directory
# $4: start run
# $5: end run
# $6: constraints_id
# $7: bdp_multiplier
# $8: x2

if [ "$#" -ne 8 ]; then
  echo "eval_orca.sh [ERROR] Only $# args found. Expected 8"
  echo "Usage: $0 <model_name> <trace_dir> <results_dir> <start_run> <end_run> <constraints_id> <x2>"
  exit 1
fi

if [[ -n "$(ls -A /mydata/log/ 2>/dev/null)" ]]; then
    read -p "[WARN] [WARN] [WARN] /mydata/log not empty - do you want to continue? (yes/no): " response
    if [[ $response != "yes" ]]; then
        exit
    fi
fi

model_name=$1
trace_dir=$2
results_dir=$3
start_run=$4
end_run=$5
constraints_id=$6
bdp_multiplier=$7
x2=$8

if [[ -z "$EVAL_C3_THRESHOLD" ]]; then
    threshold_unscaled=`echo $model_name | grep -o  "_threshold[0-9]*_" | sed 's/threshold//g;s/_//g'`
    if [[ -z "$threshold_unscaled" ]]; then
        echo "[ERROR] Could not infer threshold from model name and EVAL_C3_THRESHOLD is not set."
        exit 1
    fi
    echo "[INFO] Inferring threshold=$threshold from model name"
else
    threshold_unscaled="$EVAL_C3_THRESHOLD"
    echo "[INFO] setting threshold=$threshold_unscaled from env variable"
fi

threshold=$(bc <<< "scale=2 ; $threshold_unscaled/100.0")
echo "[INFO] Using threshold=$threshold"


# deep buffer = 11
if [[ "$constraints_id" == 11 ]]; then
    if (( $(echo "$bdp_multiplier < 2" | bc -l) )); then
        echo "[ERROR] why are you evaluating deep buffer model with such a small qsize?"
        exit 1
    fi
fi

# shallow buffer = 12
if [[ "$constraints_id" == 12 ]]; then
    if (( $(echo "$bdp_multiplier > 2" | bc -l) )); then
        echo "[ERROR] why are you evaluating shallow buffer model with such a large qsize?"
        exit 1
    fi
fi

# results_dir=$results_dir/constraints_id_$constraints_id/BDP_$bdp_multiplier/x2_$x2/k_$k/
echo "Will save results to $results_dir. Will create if not existing"
mkdir -p $results_dir

eval_k_steps=1
eval_k_symbolic_components=25
reward_mode="raw-sym"

# Running only for 1 epochs for quick evaluation, followed by 1500 epochs for symbolic reward calculation.
max_eval_actor_epochs=1

# Use seed 0 for eval.
seed=0

fixed_bw_traces=("wired6" "wired12" "wired24" "wired48" "wired96" "wired192")

# Construct the runs array.
runs=($(seq $start_run $end_run))

# Execute the experiments 5 times.
for run in ${runs[@]}
do
    # Iterate over all traces in the trace directory.
    for trace in $(ls $trace_dir)
    do
        # Check if the trace is a file.
        if [ -f $trace_dir/$trace ]
        then
            # If the trace is a fixed bandwidth trace, then run only if it is in the fixed_bw_traces list.
            if [[ "$trace_dir" != *"sage_traces"* ]]; then
                if [[ $trace == *"wired"* ]]
                then
                    if [[ ! " ${fixed_bw_traces[@]} " =~ " ${trace} " ]]
                    then
                        echo "Skipping $trace. Fixed bandwidth trace not in traces of concern."
                        continue
                    fi
                fi
            else
                # if SAGE traces
                if [[ $trace == *"wired24000"* || $trace == *"wired48000"* ]]; then
                    echo "Skipping $trace. We don't care about this one."
                    continue
                fi
            fi
            
            echo "[INFO] Running evaluation for $trace."

            dl=`echo $trace | grep -oP 'wired[0-9]*\.?[0-9]+' | sed 's/wired//g'`
            del=10
            bdp=$(echo "2 * $dl * $del / 12.0" | bc -l)
            qs=$(echo "$bdp_multiplier * $bdp" | bc -l | awk '{print int($1)}')
            
            # logfile_name="orca-$scheme-$down-$up-$latency-${period}-$qsize-${model_name}-${constraints_id}-${threshold}-${seed}"
            logfile_name="orca-cubic-$trace-wired192-10-20-$qs-${model_name}-${constraints_id}-${threshold}-0"

            if [[ -e /mydata/log/down-${logfile_name} ]]; then
                echo "Trace $trace already processed. Skipping [/mydata/log/down-${logfile_name}]"
                continue
            else
                echo "Trace $trace not exists. Will run now."
            fi

            # Run evaluation.
            C3_TRACE_DIR=$trace_dir ./scripts/run_eval.sh \
                $model_name \
                $trace \
                $eval_k_steps \
                $eval_k_symbolic_components \
                $reward_mode \
                $max_eval_actor_epochs \
                $seed \
                $constraints_id \
                $threshold \
                $x2 \
                $bdp_multiplier

            echo "[INFO] Evaluation completed for $trace. Sleeping 5 seconds."
            sleep 5
        else
            echo "Skipping $trace. Not a file."
        fi
    done

    # Copy result files from /mydata/log to the results directory.
    mkdir -p ${results_dir}/run${run}
    mv /mydata/log/* ${results_dir}/run${run}/
    rm -rfv /mydata/log*

    # Also move the rl-module/evaluation_log/seed0 directory to the results directory.
    mv ./rl-module/evaluation_log/seed0 ${results_dir}/run${run}/evaluation_log
    rm -rfv ./rl-module/evaluation_log/seed0
done
