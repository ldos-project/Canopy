lambda_=$1
k_symbolic_components=$2
k=$3

./scripts/v9_run_multi_only_heu.sh $lambda_ $k_symbolic_components $k
./scripts/v9_run_multi_only_heu_eval.sh $lambda_ $k_symbolic_components $k
