# ITHOR_Navi

## Setup
```
git clone https://github.com/jeongeun980906/ITHOR_Navi
cd ITHOR_NAVI
git submodule update --init --recursive
mkdir res
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
cd osod
python setup.py install
```

### Detector Module

Please get pretrained checkpoints on this repository [URL](https://github.com/jeongeun980906/Open-Set-Object-Detection).

### Co occurance Module
Please look at README.md on co_occurance folder! 

### Run Code
```
python SPL2.py
```
