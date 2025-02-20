first_bw=$1

first_time=1
port_base=44444
max_actor_epochs=10000

experiment_id="only_tcp_with_bw"${first_bw}
num_actors=$(((192-$first_bw+6)/6))
./orca_v2_multi_collect_traces.sh ${first_time} ${port_base} ${experiment_id} ${max_actor_epochs} ${num_actors} ${first_bw}
first_bw=$(($first_bw+48))
experiment_id="only_tcp_with_bw"${first_bw}
num_actors=$(((192-$first_bw+6)/6))
./orca_v2_multi_collect_traces.sh ${first_time} ${port_base} ${experiment_id} ${max_actor_epochs} ${num_actors} ${first_bw}
first_bw=$(($first_bw+48))
experiment_id="only_tcp_with_bw"${first_bw}
num_actors=$(((192-$first_bw+6)/6))
./orca_v2_multi_collect_traces.sh ${first_time} ${port_base} ${experiment_id} ${max_actor_epochs} ${num_actors} ${first_bw}
first_bw=$(($first_bw+48))
experiment_id="only_tcp_with_bw"${first_bw}
num_actors=$(((192-$first_bw+6)/6))
./orca_v2_multi_collect_traces.sh ${first_time} ${port_base} ${experiment_id} ${max_actor_epochs} ${num_actors} ${first_bw}
