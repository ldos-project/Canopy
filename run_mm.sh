logname=$1
port=$2
ip=`echo $SSH_CLIENT | cut -d ' ' -f1`

sudo -u $USER mm-delay 0 mm-link /home/$USER/ConstrainedOrca/wired192 /home/$USER/ConstrainedOrca/wired192 --downlink-log=/home/$USER/logs/down-${logname} -- sh -c "sudo -u $USER /home/$USER/ConstrainedOrca/client $ip 1 $port" &