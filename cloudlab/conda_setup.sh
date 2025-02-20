exit 1 # WARN: untested script
cd $HOME 
rm -rf ~/venv/ ~/Miniconda3-latest-Linux* /users/${USER}/miniconda3/
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
eval "$(/users/$USER/miniconda3/bin/conda shell.bash hook)"
conda create -n tf14_env tensorflow=1.14 -y
conda init

conda activate tf14_env

pip install --upgrade pip
pip install gym sysv_ipc
pip install git+https://github.com/deepmind/interval-bound-propagation
pip install cmake
pip install dm-sonnet==1.34
pip install tensorflow-probability==0.7.0
pip install scikit-learn

mkdir -p ~/venv/bin/
ln -sf /users/`logname`/miniconda3/envs/tf14_env/bin/python3 ~/venv/bin/python 
ln -sf /users/`logname`/miniconda3/envs/tf14_env/bin/python3 ~/venv/bin/python3

bash ~/ConstrainedOrca/build_v2.sh