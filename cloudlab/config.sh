#!/usr/bin/env bash
# Arguments:
# 1: Name of the experiment
# 2: Start node
# 3: End node
# 4: cluster
# 5: cloudlab username
# 6 (optional): KERNEL

# Check if there are atleast 5 arguments
if [[ $# -lt 5 ]]; then
  echo "Usage (option 1): $0 <experiment_name> <start_node> <end_node> <CLOUDLAB_CLUSTER> <CLOUDLAB_USERNAME>"
  echo "OR"
  echo "Usage (option 2): $0 <experiment_name> <start_node> <end_node> <CLOUDLAB_CLUSTER> <CLOUDLAB_USERNAME> KERNEL"
  echo "Use option 1 when you are using the VerifiedMLSys:orca profile on cloudlab, which has linux-learner kernel already installed"
  echo "Use option 2 when you want this script to install the linux-learner kernel"
  exit 1
fi

declare -A clusternames
clusternames[emu]="emulab.net"
clusternames[wisc]="wisc.cloudlab.us"
clusternames[utah]="utah.cloudlab.us"

if [[ ! -v clusternames[$4] ]]; then
  echo "'$4' is not a valid cluster."
  # You can exit or continue as needed
  exit 0
fi

HOSTS=`./cloudlab/nodes.sh $1 $2 $3 ${clusternames[$4]} $5 --all | tr -d ' ' | xargs`
echo "Hosts: $HOSTS"
all_ips=""
for host in $HOSTS; do
    hostname_only=`echo $host | cut -d '@' -f2`
    this_ip=`ping -c 1 $hostname_only | grep -Eo '([0-9]{1,3}\.){3}[0-9]{1,3}' | sort -u`
    
    if [[ -z "$this_ip" || $(echo "$this_ip" | wc -l) -gt 1 ]]; then
      echo "Was unable to procure IP address for $host via ping. Trying SSH + ifconfig."
      this_ip=`ssh -o StrictHostKeyChecking=no $host "ifconfig | grep -C4 eno1 | grep -v inet6 | grep inet | tr -s ' ' | cut -d ' ' -f3"`  
      if [[ -z "$this_ip" ]]; then
        echo "Fatal error. SSH+ifconfig also failed to fetch IP"  
        exit 1
      fi
      exit 1
    fi
    all_ips="$all_ips $this_ip"
done
echo "IPs: $all_ips"

echo "Configuring public keys for first node"
i=0
for host in $HOSTS; do
  echo $host
  if [ $i -eq 0 ] ; then
    echo "Test"
    ssh -o StrictHostKeyChecking=no $host "ssh-keygen"
    pkey=`ssh -o StrictHostKeyChecking=no $host "cat ~/.ssh/id_rsa.pub"`
    let i=$i+1
    continue
  fi
  ssh -o StrictHostKeyChecking=no $host "echo -e \"$pkey\" >> ~/.ssh/authorized_keys"
done

./cloudlab/copy_files.sh "$HOSTS" $5

# Increase space on the nodes
for host in $HOSTS ; do
  echo "Increasing space on host $host"
  ssh -o StrictHostKeyChecking=no $host "tmux new-session -d -s config \"
    sudo mkdir -p /mydata &&
    sudo /usr/local/etc/emulab/mkextrafs.pl -f /mydata &&

    pushd /mydata/local &&
    sudo chmod 775 -R . &&
    popd\""
done

for host in $HOSTS; do
  echo "After mkextrafs: waiting for $host tmux to finish"
  ssh $host 'while tmux has-session -t config 2>/dev/null; do sleep 1; done'
done

# Setup kernel patches
if [[ $6 == "KERNEL" ]]; then
  for host in $HOSTS; do
    echo "Updating kernel on $host ..."
    ssh -o StrictHostKeyChecking=no $host "./kernel_update.sh 2>&1" &
  done
  wait

  # Wait for the nodes to reboot
  sleep 1m
  echo "Waiting for nodes to reboot ..."

  # Check if the nodes are reachable via a SSH command every 1 minute
  while [ 1 ]; do
    FLAG=0
    for host in $HOSTS; do
      HOSTNAME=$(echo $host | awk -F'@' '{print $2}')
      nc -zw 1 $HOSTNAME 22 > /dev/null
      OUT=$?
      if [ $OUT -eq 1 ] ; then
        echo "Waiting for $host to come up ..."
        FLAG=1
        sleep 1m
      fi
    done
    if [ $FLAG -eq 0 ]; then
      break
    fi
  done
fi 

if [[ $6 == "KERNEL" ]]; then
  for host in $HOSTS ; do
    echo "Configuring Linux dependencies for $host"
    ssh -o StrictHostKeyChecking=no $host "tmux new-session -d -s config \"
      cd \$HOME &&
      sudo apt-get update &&

      sudo apt install -y build-essential git debhelper autotools-dev dh-autoreconf iptables &&
      sudo apt install -y protobuf-compiler libprotobuf-dev pkg-config libssl-dev gnuplot &&
      sudo apt install -y dnsmasq-base ssl-cert libxcb-present-dev libcairo2-dev iproute2 &&
      sudo apt install -y libpango1.0-dev iproute2 apache2-dev apache2-bin dnsmasq-base apache2-api-20120211 libwww-perl &&


      wget https://github.com/Kitware/CMake/releases/download/v3.20.1/cmake-3.20.1.tar.gz &&
      tar xzf cmake-3.20.1.tar.gz &&
      pushd cmake-3.20.1 &&
      ./bootstrap --parallel=\$(getconf _NPROCESSORS_ONLN) &&
      make -j\$(getconf _NPROCESSORS_ONLN) &&
      sudo make install &&
      export PATH=\"/usr/local/bin:\$PATH\" &&
      popd &&

      git clone https://github.com/ravinet/mahimahi && 
      pushd mahimahi &&
      ./autogen.sh && ./configure && make &&
      sudo make install &&
      sudo sysctl -w net.ipv4.ip_forward=1 &&
      popd\"" &
  done
  for host in $HOSTS; do
    echo "After kernel build: waiting for $host tmux to finish"
    ssh $host 'while tmux has-session -t config 2>/dev/null; do sleep 1; done'
  done
fi
wait



for host in $HOSTS ; do
  echo "Configuring Python dependencies for $host"
  ssh -o StrictHostKeyChecking=no $host "tmux new-session -d -s config \"
    cd \$HOME &&
    rm -rfv ~/venv/ &&
    mkdir ~/venv &&
    sudo apt update &&
    sudo apt install -y python3-pip &&
    sudo pip3 install -U virtualenv &&
    virtualenv ~/venv -p python3 &&
    source ~/venv/bin/activate &&
    pip install --upgrade pip &&
    pip install gym tensorflow==1.14 sysv_ipc &&
    pip install git+https://github.com/deepmind/interval-bound-propagation &&
    pip install cmake &&
    pip install dm-sonnet==1.34 &&
    pip install tensorflow-probability==0.7.0 &&
    pip install scikit-learn &&

    ./ConstrainedOrca/build_v2.sh\"" &
done
wait
for host in $HOSTS; do
  echo "After Python build: waiting for $host tmux to finish"
  ssh $host 'while tmux has-session -t config 2>/dev/null; do sleep 1; done'
done


if [[ $4 == "wisc" ]]; then
  for host in $HOSTS ; do
    ssh -o StrictHostKeyChecking=no $host "sudo ln -s /proj/verifiedmlsys-PG0/ /proj/VerifiedMLSys"
  done
fi

for host in $HOSTS ; do
  ssh -o StrictHostKeyChecking=no $host "mkdir -p ~/actor_logs/"
done

for host in $HOSTS ; do
  ssh -o StrictHostKeyChecking=no $host "
    if [ -d /proj/VerifiedMLSys/ConstrainedOrca/traces ]; then
      echo '$host: Trace directory exists'
    else
      echo '$host: Trace directory does not exist at /proj/VerifiedMLSys/ConstrainedOrca/traces'
    fi"
done

echo "Done setting up"
echo "To update code files on nodes do ./cloudlab/copy_files.sh \"${HOSTS[@]}\" build;"
echo "To generate params file, run the following cmd on node0: ./cloudlab/setup_params.sh \"${all_ips}\" <n_actors_per_host>;"
echo "In the setup_params.sh file, if you do not want node0 to have any actors, omit it from the list of hosts provided as CL args;"