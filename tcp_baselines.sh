# Arguments:
# $1: trace directory
# $2: results directory
# $3: tcp algo to use
# $4: bdp_multiplier
# $5: start run
# $6: end run

if [ "$#" -ne 6 ]; then
  echo "./tcp_baselines.sh [ERROR] Only $# args found. Expected 6"
  echo "Usage: $0 <trace_dir> <results_dir> <tcp_algo> <bdp_multiplier> <start_run> <end_run>"
  exit 1
fi

if [[ -n "$(ls -A /mydata/log/ 2>/dev/null)" ]]; then
    read -p "[WARN] [WARN] [WARN] /mydata/log not empty - do you want to continue? (yes/no): " response
    if [[ $response != "yes" ]]; then
        exit
    fi
fi

trace_dir=$1
results_dir=$2
tcp_algo_kernel=$3
bdp_multiplier=$4
start_run=$5
end_run=$6

case "$tcp_algo_kernel" in
  cubic|vegas|bbr|reno)
    echo "TCP algorithm is valid: $tcp_algo_kernel"
    ;;
  *)
    echo "Invalid TCP algorithm: $tcp_algo_kernel"
    exit 1
    ;;
esac

del=10 
ul=192
uplink="wired$ul"
runs=($(seq $start_run $end_run))

# Code to evaluate over *all* downlink traces. Changed to evaluation over a few.
# dl_array=`ls -p /proj/VerifiedMLSys/ConstrainedOrca/traces | grep -v  "/" | grep -v "-" | xargs`

# real_world_links=`ls /proj/VerifiedMLSys/ConstrainedOrca/traces/variable-links/`
# real_world_links_processed=""

# for item in $real_world_links; do
#   real_world_links_processed="$real_world_links_processed variable-links/$item"
# done

# dl_array="$dl_array $real_world_links_processed"

fixed_bw_traces=("wired6" "wired12" "wired24" "wired48" "wired96" "wired192")

# Constants used by tcp_eval.
port_base=44444

pushd ~/ConstrainedOrca
    sudo sysctl -w net.ipv4.tcp_congestion_control=$tcp_algo_kernel

    for run in ${runs[@]}; do
        echo "Run $run"

        # Duplicate code from eval_orca.sh to find the traces of interest.
        for downlink in $(ls $trace_dir)
        do
            tcp_algo=`sysctl net.ipv4.tcp_congestion_control | cut -d '=' -f2 | tr -d ' '`
            
            # Check if the trace is a file.
            if [ -f $trace_dir/$downlink ]
            then
                # If the trace is a fixed bandwidth trace, then run only if it is in the fixed_bw_traces list.
                
                # Use a default of 30Mbps for dl. Most time steps are less than 30Mbps in the variable link traces. For other traces, extract from the name.
                dl=30
                if [[ "$trace_dir" == *"sage_traces"* ]]; then
                    if [[ $downlink == *"wired24000"* || $downlink == *"wired48000"* ]]; then
                        echo "[SAGE] Skipping $trace. We don't care about this one."
                        continue
                    fi
                    dl=`echo $downlink | grep -oP 'wired[0-9]*\.?[0-9]+' | sed 's/wired//g'`
                elif [[ $downlink == *"wired"* ]]
                then
                    if [[ ! " ${fixed_bw_traces[@]} " =~ " ${downlink} " ]]
                    then
                        echo "Skipping $downlink. Fixed bandwidth trace not in traces of concern."
                        continue
                    fi
                    dl=`echo $downlink | grep -oP '\d+'`
                elif [[ $downlink == *"bump"* ]] || [[ $downlink == *"sawtooth"* ]] || [[ $downlink == *"mountain"* ]]
                then
                    dl=`echo $downlink | grep -oP '\d+$'`
                fi

                if [[ -z $dl ]]; then
                    echo "Error: dl is empty!" >&2
                    exit 1
                fi

                echo -e "Starting run with trace=${downlink//\//_}, dl=${dl}, ul=$ul with algo $tcp_algo"

                # Use qsize either 2*BDP or 10*BDP.
                bdp=$(echo "2 * $dl * $del / 12.0" | bc -l)
                qsize=$(echo "$bdp_multiplier * $bdp" | bc -l | awk '{print int($1)}')
                
                log_file="tcp_${tcp_algo}-${bdp_multiplier}bdp-${downlink//\//_}-${ul}-${del}-${qsize}"
                
                if [ -f "${results_dir}/tcp_${tcp_algo_kernel}/run${run}/down-$log_file" ] || [ -f "/mydata/log/down-$log_file" ]; then
                    echo -e "    Skipping. File already exists."
                    continue
                fi

                # Run the TCP evaluation binary.
                cur_dir=`pwd -P`
                path="${cur_dir}/rl-module"
                echo "Starting"
                # Runs for 150 seconds and then stops.
                C3_TRACE_DIR=$trace_dir ${path}/tcp_eval ${port_base} ${path} ${tcp_algo} ${downlink} ${uplink} ${del} ${log_file} 150 ${qsize}

                # MAHIMAHI_BASE=10.64.0.1 sudo -u `whoami` mm-delay $del mm-link "/proj/VerifiedMLSys/ConstrainedOrca/traces/$uplink" "/proj/VerifiedMLSys/ConstrainedOrca/traces/${downlink}" --downlink-log=/mydata/log/down-$log_file --uplink-queue=droptail --uplink-queue-args="packets=$qsize" --downlink-queue=droptail --downlink-queue-args="packets=$qsize" -- sh -c 'sudo -u `whoami` ./src/client $MAHIMAHI_BASE 1 44444' &

                echo "[INFO] Evaluation completed for $downlink. Doing some analysis."
                out="sum-${log_file}.tr"
                echo ${log_file} >> /mydata/log/$out
                $path/mm-thr 500 /mydata/log/down-${log_file} 1>tmp_figure 2>res
                cat res >>/mydata/log/$out
                rm res tmp_figure

                sleep 5
            else
                echo "Skipping $downlink. Not a file."
            fi
        done

        # Move the result files from /mydata/log to the results directory.
        mkdir -p ${results_dir}/tcp_${tcp_algo_kernel}/run${run}/
        mv /mydata/log/* ${results_dir}/tcp_${tcp_algo_kernel}/run${run}/
    done
popd

sudo sysctl -w net.ipv4.tcp_congestion_control=cubic # reset it after everything is done