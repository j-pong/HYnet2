# HYnet
Fully Unsupervised Learning for Continual Sequence when High Local Correlation

## Installation


### 1. Installaion of docker for ssh env
- Basic of docker installation
https://colab.research.google.com/drive/1YhIBX9i59RN_9HEMihJX6TnFm9G5a7UL?authuser=1#scrollTo=swQ7g70S9O4J

- Docker for HYnet
```bash
cd docker

make _build

make run
```

### 2. Installaion of kaldi for speech
```bash
git clone https://github.com/kaldi-asr/kaldi kaldi

cd kaldi/tools
extras/install_mkl.sh -s
extras/check_dependencies.sh
make -j 28
extras/install_irstlm.sh

cd ../src/
./configure
make depend -j 28
make -j 28
```

### 3. Installaion of espnet for input pipelines
```bash
cd tools

./meta_installers/install_espnet.sh
```

### 4. Installaion of hynet for customizing egs
```bash
cd tools

./meta_installers/install_hynet.sh
```

## Make custom example with task
```bash
cp -r tools/espnet/egs2/TEMPLATE/asr1/* egs/TEMPLATE/asr1/
```

## Bugfix

ctc_segmentation/ctc_segmentation_dyn.pyx error
- Remove ctc_segmentation at tools/espnet/setup.py that is included in requirements

ctc install error with pip version
```bash
pip install pip==19; pip install warpctc-pytorch==0.2.1+torch16.cuda102
```

matplotlib version error
- Remove matplotlib at tools/espnet/setup.py that is included in requirements