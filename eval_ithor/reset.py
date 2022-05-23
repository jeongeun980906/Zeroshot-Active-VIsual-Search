# importing sys
import sys

# adding Folder_2 to the system path
sys.path.insert(0, '/home/jeongeun/test_env/Open-Set-Object-Detection')

# Detector module
from data.phase_1 import load_voc_instances,VOC_CLASS_NAMES
import torch
import math
import matplotlib.pyplot as plt
from structures.box import Boxes
from engine.predictor import DefaultPredictor

from  config.config import get_cfg
from model.rcnn import GeneralizedRCNN
from detector.postprocess import postprocess,plot_openset,plot_candidate

## Matching Module
from detector.query_matching import matcher
def reset_scene(controller):
    controller.reset(
    # makes the images a bit higher quality
    width=800,
    height=800,

    # Renders several new image modalities
    renderDepthImage=True,
    renderClassImage = True,
    renderSemanticSegmentation=False,
    renderNormalsImage=False
    )
    scene_bounds = controller.last_event.metadata['sceneBounds']['center']
    controller.step(
        action="AddThirdPartyCamera",
        position=dict(x=scene_bounds['x'], y=5.0, z=scene_bounds['z']),
        rotation=dict(x=90, y=0, z=0),
        orthographic=True,
        orthographicSize= 5.0, fieldOfView=100,
        skyboxColor="white"
    )
    controller.step(dict(action='GetReachablePositions'))
    rstate = controller.last_event.metadata['actionReturn']

    controller.step(
        action="Teleport",
        position = rstate[100]
    )
    objects = controller.last_event.metadata['objects']
    return objects, rstate
    

def load_detector(device='cuda:0',ID=19):
    '''
    config file
    '''
    cfg = get_cfg()
    cfg.merge_from_file('../Open-Set-Object-Detection/config_files/voc.yaml')
    cfg.MODEL.SAVE_IDX=ID
    cfg.MODEL.RPN.USE_MDN=False
    cfg.log = False 
    cfg.MODEL.ROI_HEADS.USE_MLN = True
    cfg.MODEL.ROI_HEADS.AUTO_LABEL = False
    cfg.MODEL.ROI_HEADS.AF = 'baseline'
    cfg.MODEL.RPN.AUTO_LABEL = False
    cfg.MODEL.ROI_BOX_HEAD.USE_FD = False
    cfg.MODEL.RPN.AUTO_LABEL_TYPE = 'sum'
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 21
    cfg.INPUT.RANDOM_FLIP = "none"
    cfg.MODEL.ROI_HEADS.UNCT = True
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3
    cfg.PATH = '../Open-Set-Object-Detection'

    # cfg.merge_from_list(args.opts)
    RPN_NAME = 'mdn' if cfg.MODEL.RPN.USE_MDN else 'base'
    ROI_NAME = 'mln' if cfg.MODEL.ROI_HEADS.USE_MLN else 'base'
    MODEL_NAME = RPN_NAME + ROI_NAME
    # cfg.merge_from_list(args.opts)
    cfg.freeze()
    # wandb.init(config=cfg,tags= 'temp',name = 'temp',project='temp')

    model = GeneralizedRCNN(cfg,device = device).to(device)
    state_dict = torch.load('../Open-Set-Object-Detection/ckpt/{}/{}_{}_15000.pt'.format(cfg.MODEL.ROI_HEADS.AF,cfg.MODEL.SAVE_IDX,MODEL_NAME),map_location=device)
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
    model.load_state_dict(pretrained_dict)

    predictor = DefaultPredictor(cfg,model)
    return predictor


def load_detector_base(device='cuda:0'):
    '''
    config file: VOC trained only
    '''
    cfg = get_cfg()
    cfg.merge_from_file('../Open-Set-Object-Detection/config_files/voc.yaml')
    cfg.MODEL.SAVE_IDX=2
    cfg.MODEL.RPN.USE_MDN=False
    cfg.log = False 
    cfg.MODEL.ROI_HEADS.USE_MLN = False
    cfg.MODEL.ROI_HEADS.AUTO_LABEL = False
    cfg.MODEL.ROI_HEADS.AF = 'baseline'
    cfg.MODEL.RPN.AUTO_LABEL = False
    cfg.MODEL.ROI_BOX_HEAD.USE_FD = False
    cfg.MODEL.RPN.AUTO_LABEL_TYPE = 'sum'
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20
    cfg.INPUT.RANDOM_FLIP = "none"
    cfg.MODEL.ROI_HEADS.UNCT = False
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3
    cfg.PATH = '../Open-Set-Object-Detection'

    # cfg.merge_from_list(args.opts)
    RPN_NAME = 'mdn' if cfg.MODEL.RPN.USE_MDN else 'base'
    ROI_NAME = 'mln' if cfg.MODEL.ROI_HEADS.USE_MLN else 'base'
    MODEL_NAME = RPN_NAME + ROI_NAME
    # cfg.merge_from_list(args.opts)
    cfg.freeze()
    # wandb.init(config=cfg,tags= 'temp',name = 'temp',project='temp')

    model = GeneralizedRCNN(cfg,device = device).to(device)
    state_dict = torch.load('../Open-Set-Object-Detection/ckpt/{}/{}_{}_15000.pt'.format(cfg.MODEL.ROI_HEADS.AF,cfg.MODEL.SAVE_IDX,MODEL_NAME),map_location=device)
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
    model.load_state_dict(pretrained_dict)

    predictor = DefaultPredictor(cfg,model)
    return predictor

def get_min_dis(query_object,controller,map,schedular):
    gt_landmark_ID = query_object['parentReceptacles']
    if gt_landmark_ID != None:
        for e,l in  enumerate(map.landmarks):
            for ids in l['ID']:
                if ids in gt_landmark_ID:
                    min_loi = map.landmark_loi[e][0]
                    cpos = controller.last_event.metadata['agent']['position']
                    min_dis = schedular.shortest_path_length(controller,min_loi[0],cpos)
                    return min_dis
    query_pos = query_object['position']
    
    l_dis=[]
    for l in map.landmarks:
        l_pos = l['cp']
        dis = math.sqrt((query_pos['x']-l_pos['x'])**2+(query_pos['z']-l_pos['z'])**2)
        l_dis.append(dis)
    min_dis = min(l_dis)
    min_index = l_dis.index(min_dis)
    min_loi = map.landmark_loi[min_index][0]
    cpos = controller.last_event.metadata['agent']['position']
    min_dis = schedular.shortest_path_length(controller,min_loi[0],cpos)
    return min_dis

def move_init(controller,rstate):
    controller.step(
        action="Teleport",
        position = rstate[100]
    )