# install conda
rm Miniconda3-py312_25.9.1-3-Linux-x86_64.sh
# assumes Python 3.12 on host !!!
wget https://repo.anaconda.com/miniconda/Miniconda3-py312_25.9.1-3-Linux-x86_64.sh
bash Miniconda3-py312_25.9.1-3-Linux-x86_64.sh -u -b -p ./miniconda3
source miniconda3/bin/activate

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda init --all

# conda env ma
conda create -y -n ma
conda activate ma

# conda install python
conda install -y -c conda-forge python=3.14

# conda install jupyter
conda install -y -c conda-forge jupyterlab==4.5.0
conda install -y -c conda-forge notebook==7.5.0

# venv ma
mkdir -p ./venvs
python3 -m venv ./venvs/ma
source ./venvs/ma/bin/activate
pip3 install -U pip build

./xray_fov/build_and_install.sh
./xray_fov/install_dev.sh

pip3 install ipykernel
python3 -m ipykernel install --user --name pytorch --display-name "pytorch"

rm Miniconda3-py312_25.9.1-3-Linux-x86_64.sh