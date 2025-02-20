#!/usr/bin/env bash
# Arguments:
# 1: Name of the experiment
# 2: Start node of experiment
# 3: End node of experiment
# 4: CLOUDLAB_CLUSTER
# 5: CLOUDLAB_USERNAME
# 6: -a/--all to denote that all nodes have to be used

EXP_NAME=$1
NUM_NODE=$3
NODE_PREFIX="node"

DOMAIN=$4
USER_NAME=$5

if [ -z "${DOMAIN}" ] || [ -z "${USER_NAME}" ]; then
  echo "Error: DOMAIN ($DOMAIN) or USER_NAME ($USER_NAME) is not set or is empty"
  exit 1
fi

if [[ "$DOMAIN" == "wisc.cloudlab.us" || "$DOMAIN" == "utah.cloudlab.us" ]]; then
  PROJECT_EXT="verifiedmlsys-PG0"
elif [[ "$DOMAIN" == "emulab.net" ]]; then
  PROJECT_EXT="verifiedmlsys"
else
  echo "Invalid DOMAIN: $DOMAIN"
  exit 1
fi

# Some nodes' NICs might constantly fail, set SKIP_NODES to skip those nodes
# SKIP_NODES="7"
SKIP_NODES=""

# Manually override the order in which the nodes are listed
# SKIP_NODES and -a (--all) will not work
MANUAL_ORDER=""

if [ -z "$MANUAL_ORDER" ]; then
  i=$2
  while [ $i -le $NUM_NODE ]; do
    skip=0
    if [ "$6" != "-a" ] && [ "$6" != "--all" ]; then
      for node in $SKIP_NODES; do
        if [ "$i" -eq "$node" ]; then
          skip=1
          break
        fi
      done
    fi

    if [ $skip -eq 0 ]; then
      echo "$USER_NAME@$NODE_PREFIX$i.$EXP_NAME.$PROJECT_EXT.$DOMAIN"
    fi

    let i=$i+1
  done
else
  for i in $MANUAL_ORDER; do
    echo "$USER_NAME@$NODE_PREFIX$i.$EXP_NAME.$PROJECT_EXT.$DOMAIN"
  done
fi
