#!/bin/bash

# Clone the espnet
rm -rf espnet
git clone https://github.com/espnet/espnet

# HYnet is extension of espnet
cp -r espnet/tools/* ./

# The espnet installation process https://espnet.github.io/espnet/installation.html
. ./setup_cuda_env.sh /usr/local/cuda
. ./setup_anaconda.sh anaconda base 3.8

# We needs just pyotrch if needs more dependency then install like this
. ./activate_python.sh && conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
touch pytorch.done

. ./activate_python.sh && ./installers/install_fairseq.sh
touch fairseq.done

./installers/install_sctk.sh
touch sctk.done

# The espent editable import to HYnet
. ./activate_python.sh && cd espnet/tools && python3 -m pip install -e "..[recipe]"
touch espnet.done