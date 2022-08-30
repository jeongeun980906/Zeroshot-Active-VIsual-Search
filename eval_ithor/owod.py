from ithor_tools.landmark_utils import get_gt_box,in_landmark_names,out_landmark_names,landmark_names
import sys
sys.path.append("..") 
import cv2
import os
import torch
from torch.distributions.weibull import Weibull
from torch.distributions.transforms import AffineTransform
from torch.distributions.transformed_distribution import TransformedDistribution

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


def load_OWOD(device='cuda:0'):
    model = '/home/jeongeun/OWOD/output/t1_final/model_final.pth'
    cfg_file = '/home/jeongeun/OWOD/configs/OWOD/t1/t1_test.yaml'
    # Get the configuration ready
    cfg = get_cfg()
    cfg.merge_from_file(cfg_file)
    cfg.MODEL.WEIGHTS = model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    # cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.8
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3
    # cfg.MODEL.ROI_HEADS.NUM_CLASSES = 21

    predictor = DefaultPredictor(cfg)

    param_save_location = os.path.join('/home/jeongeun/OWOD/output/t1_final/energy_dist_' + str(20) + '.pkl')
    params = torch.load(param_save_location)
    unknown = params[0]
    known = params[1]
    #print(known,unknown)
    unk_dist = create_distribution(unknown['scale_unk'], unknown['shape_unk'], unknown['shift_unk'])
    known_dist = create_distribution(known['scale_known'], known['shape_known'], known['shift_known'])
    return predictor,unk_dist,known_dist

def create_distribution(scale, shape, shift):
    wd = Weibull(scale=scale, concentration=shape)
    transforms = AffineTransform(loc=shift, scale=1.)
    weibull = TransformedDistribution(wd, transforms)
    return weibull


def compute_prob(x, distribution):
    eps_radius = 0.5
    num_eval_points = 100
    start_x = x - eps_radius
    end_x = x + eps_radius
    step = (end_x - start_x) / num_eval_points
    dx = torch.linspace(x - eps_radius, x + eps_radius, num_eval_points)
    pdf = distribution.log_prob(dx).exp()
    prob = torch.sum(pdf * step)
    return prob


def update_label_based_on_energy(logits, classes, unk_dist, known_dist):
    unknown_class_index = 80
    cls = classes
    lse = torch.logsumexp(logits[:, :5], dim=1).cpu()
    for i, energy in enumerate(lse):
        p_unk = compute_prob(energy, unk_dist)
        p_known = compute_prob(energy, known_dist)
        # print(str(p_unk) + '  --  ' + str(p_known))
        if torch.isnan(p_unk) or torch.isnan(p_known):
            continue
        if p_unk > p_known:
            cls[i] = unknown_class_index
    return cls

def gather_owod(controller,query_object,opos,predictor,unk_dist,known_dist,clip_matcher,
                    detection_labels,scene_memory):
    frames = []
    gt_boxes = []
    vis = 0
    pos = []
    detected_landmarks = []
    for _ in range(9):
        controller.step(action="RotateRight",degrees = 40
                            )
        pos.append(dict(pos = controller.last_event.metadata['agent']['position'], 
                    rot = controller.last_event.metadata['agent']['rotation']))
        img = controller.last_event.cv2img
        frames.append(img)
        gt_box = get_gt_box(controller,query_object,version=2,opos=opos)
        gt_boxes.append(gt_box)
        vis += 1 if gt_box != None else 0
        pred = predictor(img)
        pred = pred["instances"]
        pred_boxes = pred.pred_boxes
        pred_classes = pred.pred_classes.tolist()
        logits = pred.logits
        pred_classes = update_label_based_on_energy(logits, pred_classes, unk_dist, known_dist)
        pred_classes = torch.IntTensor(pred_classes).to(torch.device("cuda"))
        # print(pred_classes)
        in_pred_classes = []
        mask = torch.zeros_like(pred_classes,dtype=torch.bool)
        for e,cat in enumerate(pred_classes):
            if cat in detection_labels:
                mask[e] = True
                in_pred_classes.append(detection_labels.index(cat.item()))
                
        in_pred_classes = torch.LongTensor(in_pred_classes)
        in_pred_boxes = pred_boxes[mask].tensor.cpu()
        # in_pred_uncts = pred_uncts[mask].cpu()
        mask = (pred_classes == 80)
        # out_pred_uncts = pred_uncts[mask].cpu()
        out_pred_boxes = pred_boxes[mask]
        in_entropy = torch.ones_like(in_pred_classes)*0.5
        out_pred_boxes, out_pred_classes, out_entropy = clip_matcher.landmark_matching(img,out_pred_boxes)
        if len(out_pred_classes)>0:
            out_pred_classes = out_pred_classes + len(in_landmark_names)
        # print(out_pred_classes.shape,in_pred_classes.shape)
        pred_classes = torch.cat((in_pred_classes,out_pred_classes),axis=0)
        pred_boxes = torch.cat((in_pred_boxes,out_pred_boxes),axis=0)
        pred_uncts = torch.cat((in_entropy,out_entropy),axis=0)
        # print(pred_uncts)
        # plot_openset(img,pred_boxes,pred_classes,landmark_names)
        # l = m.detect_gt_box(controller)
        l = scene_memory.proj(pred_boxes,pred_classes,pred_uncts,controller,landmark_names)
        if l != None:
            detected_landmarks += l
    detected_landmarks = list(set(detected_landmarks))
    return frames,pos,gt_boxes,vis,detected_landmarks