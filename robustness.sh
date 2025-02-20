if [ "$#" -lt 5 ] || [ "$#" -gt 6 ]; then
    echo "Found $# args"
    echo "Usage: $0 <model name> <enable randomness 0/1> <idx> <std> <random method=gaussian/uniform> <file e.g. sawtooth500_96_48> [optional: name]"
    echo "If you want to run the same config multiple times, use a unique name each time"
    exit 1
fi

sudo killall -s9 python client orca-server-mahimahi_v0 orca-server-mahimahi_v2

export ROHIT_ROBUSTNESS_EXP=$2
export ROHIT_ROBUSTNESS_IDX=$3
export ROHIT_ROBUSTNESS_STD=$4
export ROHIT_ROBUSTNESS_RAN=$5

model_name=$1
trace_name=$6

if [[ "$ROHIT_ROBUSTNESS_EXP" -ne 0 && "$ROHIT_ROBUSTNESS_EXP" -ne 1 ]]; then
    echo "Error: ROHIT_ROBUSTNESS_EXP must be 0 or 1"
    exit 1
fi

if [[ "$ROHIT_ROBUSTNESS_IDX" -lt 0 || "$ROHIT_ROBUSTNESS_IDX" -gt 6 ]]; then
    echo "Error: ROHIT_ROBUSTNESS_IDX must be between 0 and 6"
    exit 1
fi

if [[ "$ROHIT_ROBUSTNESS_RAN" != "gaussian" && "$ROHIT_ROBUSTNESS_RAN" != "uniform" ]]; then
    echo "Error: ROHIT_ROBUSTNESS_RAN must be 'gaussian' or 'uniform'"
    exit 1
fi

if ls /mydata/log/down-* /mydata/log/sum-* 1> /dev/null 2>&1; then
    echo "Error: Files starting with 'down-' or 'sum-' are present in /mydata/log."
    exit 1
fi

if [ "$2" -eq 0 ]; then
    OUTPUT_FILE_SUFFIX="Orca"
else
    metric="Thr"
    if [ "$ROHIT_ROBUSTNESS_IDX" == "5" ]; then
        metric="Delay"
    fi
    OUTPUT_FILE_SUFFIX="${ROHIT_ROBUSTNESS_RAN}_${metric}_${ROHIT_ROBUSTNESS_STD}"
fi

if [ -z "$7" ]; then
    OUTPUT_FILE_SUFFIX="${OUTPUT_FILE_SUFFIX}-${7}"
fi

mkdir -p /mydata/log/${trace_name}/
OUTPUT_DOWN_FILE=/mydata/log/${trace_name}/DOWN-${OUTPUT_FILE_SUFFIX}
OUTPUT_SUM_FILE=/mydata/log/${trace_name}/SUM-${OUTPUT_FILE_SUFFIX}

if [ -e "$OUTPUT_DOWN_FILE" ] || [ -e "$OUTPUT_SUM_FILE" ]; then
    echo "Error: One or both of the output files for this config already exist."
    echo "If you want to run the same config multiple times, use a unique name as CL arg to script"
    exit 1
fi

./scripts/run_eval.sh $model_name $trace_name 1 50 raw-sym 1 0 7 ${ROHIT_ROBUSTNESS_STD} 1

for file in /mydata/log/down*; do
  if [[ -f "$file" ]]; then
    dirname=$(dirname "$file")    
    mv $file $OUTPUT_DOWN_FILE
  fi
done

for file in /mydata/log/sum*; do
  if [[ -f "$file" ]]; then
    dirname=$(dirname "$file")
    mv $file $OUTPUT_SUM_FILE
  fi
done

sudo killall -s9 python client orca-server-mahimahi_v0 orca-server-mahimahi_v2