# ITHOR_Navi

## Setup
```
mkdir ai2-detection
cd ai2-detection
git clone https://github.com/jeongeun980906/ITHOR_Navi
cd ITHOR_NAVI
mkdir res
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
cd ..
git clone https://github.com/jeongeun980906/Open-Set-Object-Detection
cd Open-Set-Object-Detection
python setup.py install
```

### Detector Module
[URL](https://github.com/jeongeun980906/Open-Set-Object-Detection)

### Co occurance Module
Please look at README.md on co_occurance folder! 

## Run
!Change Path!
**On DEMO.ipynb and eval_ithor/reset.py**
Change to
```
# importing sys
import sys
# adding Folder_2 to the system path
sys.path.insert(0, '[your path]/ai2-detection/Open-Set-Detection')
```

and also config files on DEMO.ipynb and reset.py

```
cfg = get_cfg()
cfg.merge_from_file('../Open-Set-Object-Detection/config_files/voc.yaml')
...
cfg.PATH = '../Open-Set-Object-Detection'
```
And also checkpoint path

```
device = 'cuda:1'
model = GeneralizedRCNN(cfg,device = device).to(device)
state_dict = torch.load('../Open-Set-Object-Detection/ckpt/{}/{}_{}_15000.pt'.format(cfg.MODEL.ROI_HEADS.AF,cfg.MODEL.SAVE_IDX,MODEL_NAME),map_location=device)
pretrained_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
model.load_state_dict(pretrained_dict)

predictor = DefaultPredictor(cfg,model)
```
### Run Code
```
cd ITHOR_Navi
python SPL.py
```
