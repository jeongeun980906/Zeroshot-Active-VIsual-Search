# LoI Ablation
python SPL.py --num_loi 0 --val --co_thres 0.2
python SPL.py --num_loi 1 --val --co_thres 0.2
# python SPL.py --num_loi 2
# python SPL.py --num_loi 2 --co_thres -0.3

## LoI measure
python SPL.py --num_loi 1 --dis_only --val
# python SPL.py --num_loi 2 --co_base

## Detector
# python SPL.py --num_loi 0 --base_detector --detector id 2
# python SPL.py --num_loi 2 --base_detector --dis_only --detector id 2
