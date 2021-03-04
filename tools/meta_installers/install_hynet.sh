#!/bin/bash

# The espent editable import to HYnet
. ./activate_python.sh && python3 -m pip install -e ../
. ./activate_python.sh && conda install -y torchvision captum -c pytorch

touch hynet.done
