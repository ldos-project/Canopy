link_name=$1

# models in sequence: <deep buffer model> <shallow buffer model> <robustness model> <baseline orca>
models=(
    "learner0-v9_actorNum256_multi_lambda0.25_ksymbolic5_k3_raw-sym_threshold25_seed0_constraints_id11_xtwo25"
    "learner0-v9_actorNum256_multi_lambda0.25_ksymbolic5_k1_raw-sym_threshold25_seed0_constraints_id12_xtwo25"
    # "learner0-v9_actorNum256_multi_lambda0.25_ksymbolic5_k1_raw-sym_threshold5_seed0_constraints_id7_xtwo1"
    "learner0-v9_actorNum256_multi_lambda0.0_ksymbolic5_k1_raw-sym_threshold25_seed0"
    "cubic"
    "bbr"
    # "reno"
    # "vegas"
)

cids=(11 12 11 0 0)
thresholds=(25 25 25 0 0)
x2s=(25 25 25 0 0)

case "$link_name" in
  "eastus"|"westus2"|"canada"|"southcentralus"|"europe"|"australia"|"india"|"southamerica"|"africa")
    echo "Valid region: $link_name"
    ;;
  *)
    echo "Invalid region: $link_name"
    exit 1
    ;;
esac

for run in {0..3}; do
    for i in "${!models[@]}"; do
        model="${models[$i]}"
        cid="${cids[$i]}"
        threshold="${thresholds[$i]}"
        x2="${x2s[$i]}"

        echo "Running with: model=$model, cid=$cid, threshold=$threshold, x2=$x2"
        ./scripts/real_world.sh "$model" "$link_name" "$cid" "$threshold" "$x2" "$run"
	sudo lsof -i :44444 | tr -s ' ' | cut -d ' ' -f2 | grep -v PID | xargs  kill -9
	sleep 5
    done
done
