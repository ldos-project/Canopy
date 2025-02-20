#!/usr/bin/env bash
# Arguments:
# 1: HOSTS - list of ips
# 2: n_actors_per_host

# Check if the script has two arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <HOSTS> <n_actors_per_host>"
    exit 1
fi

HOSTS=$1
n_actors_per_host=$2
LEARNER_IP_ADDR=`ifconfig | grep -C4 "en[o|p]1" | grep -v inet6 | grep inet | tr -s ' ' | cut -d ' ' -f3`

if [ -z "${LEARNER_IP_ADDR}" ]; then
    echo "Fatal error - learner IP not found :("
    exit 1
fi

if ! [[ "$n_actors_per_host" =~ ^[0-9]+$ ]] || [ "$n_actors_per_host" -le 0 ]; then
    echo "Assertion failed: n_actors_per_host should be a positive integer."
    exit 1
fi

total_actors=0
for host in $HOSTS; do
    port=33331
    for j in $(seq 1 $n_actors_per_host); do
        ACTOR_LIST+=("\"${host}:${port}\"")
        let port=$port+1
        let total_actors=$total_actors+1
    done
done

# Print the list of IP addresses
echo "List of IP addresses: ${ACTOR_LIST[@]}"

# Construct a string of comma separated IP addresses
ACTOR_STRING=$(IFS=,; echo "${ACTOR_LIST[*]}")

# adds actor, learner IPs and total actors.
sed "s/\"{ACTOR_LIST}\"/${ACTOR_STRING}/g; s/\"{N_ACTORS}\"/${total_actors}/g; s/{LEARNER_IP}/$LEARNER_IP_ADDR/g" ./cloudlab/params_distributed.template > ./rl-module/params_distributed.json