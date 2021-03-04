#!/bin/bash

rm -rf espnet
git clone https://github.com/espnet/espnet

# HYnet is extension of espnet
cp -r espnet/tools/* ./

# The espnet installation process https://espnet.github.io/espnet/installation.html
. ./setup_cuda_env.sh /usr/local/cuda
. ./setup_anaconda.sh venv base 3.8

# We needs just pyotrch if needs more dependency then install like this
make TH_VERSION=1.6 CHAINER_VERSION=6.0.0 pytorch.done

# The espent editable import to HYnet
. ./activate_python.sh && cd espnet/tools && python3 -m pip install -e "..[recipe]"

touch espnet.done