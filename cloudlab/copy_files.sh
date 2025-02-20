if [[ $# -lt 1 ]]; then
  echo "Usage (option 1): $0 '<hosts>'"
  echo "OR"
  echo "Usage (option 2): $0 '<hosts>' build"
  echo "Use option 1 when you setting up a node for the first time"
  echo "Use option 2 when you want to copy over new src files and build orca_mahimahi (only diff from option 1: this runs build_v2.sh)"
  exit 1
fi

if [[ "$1" != *"@"* ]]; then
  echo "hostnames must be of the format <cloudlab_username>@<host>."
  echo "it looks like you may have missed specifying usernames"
  exit 1
fi

TARBALL=testbed.tar.gz
tar -czf $TARBALL rl-module/ scripts/ cloudlab/ src/ *.sh *.json

HOSTS=$1

for host in $HOSTS; do
  echo "Pushing to $host ..."
  # Move the train_dir/ to a new location.
  # ssh -o StrictHostKeyChecking=no $host 'if [ -d ~/ConstrainedOrca/rl-module/train_dir ]; then mv ~/ConstrainedOrca/rl-module/train_dir ~/; else mkdir -p ~/train_dir; fi'
  ssh -o StrictHostKeyChecking=no $host "rm -rfv ~/ConstrainedOrca/; mkdir -p ~/ConstrainedOrca/;"
  scp -q -o StrictHostKeyChecking=no $TARBALL $host:~/ConstrainedOrca/$TARBALL >/dev/null 2>&1 &
  ssh -o StrictHostKeyChecking=no $host "mkdir -p ~/ConstrainedOrca/rl-module/training_log"
done
wait

for host in $HOSTS; do
  echo "Removing tar ball on $host ..."
  ssh -o StrictHostKeyChecking=no $host "cd ~/ConstrainedOrca/; tar -xzf ~/ConstrainedOrca/$TARBALL 2>&1; rm ~/ConstrainedOrca/$TARBALL" &
done
wait

rm -f $TARBALL

# Move the train_dir/ back to the original location.
# for host in $HOSTS; do
#   ssh -o StrictHostKeyChecking=no $host "mv ~/train_dir ~/ConstrainedOrca/rl-module/"
# done

if [[ $2 == "nobuild" ]]; then
  echo "[WARN WARN WARN] You have copied over files but did not build them. This is usually NOT desired behaviour"
else
  for host in $HOSTS; do
    echo "Running build_v2.sh on $host ..."
    ssh -o StrictHostKeyChecking=no $host "~/ConstrainedOrca/build_v2.sh" &
  done
  wait
fi