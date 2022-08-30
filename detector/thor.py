from ithor_tools.thor_detect import get_gt_box
from detector.postprocess import plot_openset
import torch
import numpy as np
from detector.postprocess import postprocess
from detector.postprocess import plot_openset
from ithor_tools.landmark_utils import in_landmark_names,out_landmark_names,landmark_names
try:
    from eval_ithor.owod import create_distribution,compute_prob,update_label_based_on_energy
except:
    print("No implementation for OWOD")
    pass

def gather(controller,query_object,opos,step = 4,angle = 180, show=False):
    frames = []
    gt_boxes = []
    vis = 0
    # controller.step("MoveRight")
    # for _ in range(3):
    frames.append(controller.last_event.cv2img)
    pos = [dict(pos = controller.last_event.metadata['agent']['position'], 
                    rot = controller.last_event.metadata['agent']['rotation'])]
    gt_box = get_gt_box(controller,query_object,version=2,opos=opos)
    gt_boxes.append(gt_box)
    vis += 1 if gt_box != None else 0
    controller.step("LookDown")
    pos.append(dict(pos = controller.last_event.metadata['agent']['position'], 
                rot = controller.last_event.metadata['agent']['rotation']))
    frames.append(controller.last_event.cv2img)
    gt_box = get_gt_box(controller,query_object,version=2,opos=opos)
    gt_boxes.append(gt_box)
    vis += 1 if gt_box != None else 0
    controller.step("LookUp")
    for _ in range(step-1):
        controller.step(action = "RotateRight", degrees = angle/(step-1))
        pos.append(dict(pos = controller.last_event.metadata['agent']['position'], 
                    rot = controller.last_event.metadata['agent']['rotation']))
        frames.append(controller.last_event.cv2img)
        gt_box = get_gt_box(controller,query_object,version=2,opos=opos)
        gt_boxes.append(gt_box)
        vis += 1 if gt_box != None else 0
        controller.step("LookDown")
        pos.append(dict(pos = controller.last_event.metadata['agent']['position'], 
                    rot = controller.last_event.metadata['agent']['rotation']))
        frames.append(controller.last_event.cv2img)
        gt_box = get_gt_box(controller,query_object,version=2,opos=opos)
        gt_boxes.append(gt_box)
        vis += 1 if gt_box != None else 0
        controller.step("LookUp")
    controller.step(action = "RotateLeft", degrees = angle)
    #     controller.step("MoveBack",moveMagnitude = 0.05)
    # controller.step("MoveAhead",moveMagnitude = 0.05*3)
    return frames,pos,gt_boxes,vis


def gather2(controller,query_object,predictor,postprocess,clip_matcher,
                    visible_landmark_names,scene_memory,show=False):
    frames = []
    gt_boxes = []
    vis = 0
    pos = []
    detected_landmarks = []
    for _ in range(6):
        controller.step(action="RotateRight",degrees = 60
                            )
        pos.append(dict(pos = controller.last_event.metadata['agent']['position'], 
                    rot = controller.last_event.metadata['agent']['rotation']))
        img = controller.last_event.cv2img
        frames.append(img)
        gt_box = get_gt_box(controller,query_object,version=2)
        gt_boxes.append(gt_box)
        vis += 1 if gt_box != None else 0
        pred = predictor(img)
        pred_boxes, pred_classes,_ = postprocess(pred)
        pred_boxes, pred_classes, pred_entropy = clip_matcher.landmark_matching(img,pred_boxes)
        if show:
            plot_openset(img,pred_boxes,pred_classes,visible_landmark_names)
        lnames = [visible_landmark_names[i] for i in pred_classes]
        # l = m.detect_gt_box(controller)
        l = scene_memory.proj(pred_boxes,lnames,controller,visible_landmark_names)
        if l != None:
            detected_landmarks += l
    detected_landmarks = list(set(detected_landmarks))
    return frames,pos,gt_boxes,vis,detected_landmarks

def gather3(controller,query_object,opos,predictor,postprocess,clip_matcher,
                    detection_labels,scene_memory,unk_flag=True,visualize=False):
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
        pred_boxes, pred_classes,_,pred_uncts = postprocess(pred)
        in_pred_classes = []
        mask = torch.zeros_like(pred_classes,dtype=torch.bool)
        for e,cat in enumerate(pred_classes):
            if cat in detection_labels:
                mask[e] = True
                in_pred_classes.append(detection_labels.index(cat.item()))
                
        in_pred_classes = torch.LongTensor(in_pred_classes)
        in_pred_boxes = pred_boxes[mask].tensor.cpu()
        # in_pred_uncts = pred_uncts[mask].cpu()
        mask = (pred_classes == 20)
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
        if visualize:
            plot_openset(img,pred_boxes,pred_classes,landmark_names)
        # l = m.detect_gt_box(controller)
        l = scene_memory.proj(pred_boxes,pred_classes,pred_uncts,controller,landmark_names)
        if l != None:
            detected_landmarks += l
    detected_landmarks = list(set(detected_landmarks))
    return frames,pos,gt_boxes,vis,detected_landmarks




def detect(frames,single_pos,gt_boxes,controller,predictor,matcher,d2w,unk_only_flag=True,visualize=False):
    patch = np.zeros((0,256,256,3),dtype=np.uint8)
    map_p = []
    sucesses = 0
    thres = 0.0 if unk_only_flag else 0.2
    for frame,pos,gt_box in zip(frames,single_pos,gt_boxes):
        pred = predictor(frame)
        pred_boxes, pred_classes,unk_only,_ = postprocess(pred,thres)
        if unk_only_flag:
            pred_boxes = pred_boxes[unk_only]
        show_patch,candidate_boxes,sucess = matcher.matching_score(frame,pred_boxes,gt_box)
        sucesses += torch.sum(sucess).item()
        if visualize:
            plot_openset(frame,candidate_boxes,torch.zeros((len(candidate_boxes))),[matcher.new_query_object_name])
        if len(show_patch):
            DEPTH = controller.last_event.depth_frame
            COLOR = controller.last_event.frame.astype(np.uint8)
            map_points = d2w.object_coord(candidate_boxes,DEPTH,COLOR
                                ,pos['pos'],pos['rot'])
            patch = np.concatenate((patch,show_patch),axis=0)
            map_p += map_points
    return patch,map_p,sucesses



def detect_OWOD(frames,single_pos,gt_boxes,controller,predictor,matcher,d2w,unk_dist,known_dist):
    patch = np.zeros((0,256,256,3),dtype=np.uint8)
    map_p = []
    sucesses = 0
    for frame,pos,gt_box in zip(frames,single_pos,gt_boxes):
        pred = predictor(frame)
        pred = pred["instances"]
        # dev = pred.pred_classes.get_device()
        pred_boxes = pred.pred_boxes
        classes = pred.pred_classes.tolist()
        logits = pred.logits
        classes = update_label_based_on_energy(logits, classes, unk_dist, known_dist)
        classes = torch.IntTensor(classes).to(torch.device("cuda"))
        unk_only = (classes == 80)
        pred_boxes = pred_boxes[unk_only]
        show_patch,candidate_boxes,sucess = matcher.matching_score(frame,pred_boxes,gt_box)
        sucesses += torch.sum(sucess).item()
        if len(show_patch):
            DEPTH = controller.last_event.depth_frame
            COLOR = controller.last_event.frame.astype(np.uint8)
            map_points = d2w.object_coord(candidate_boxes,DEPTH,COLOR
                                ,pos['pos'],pos['rot'])
            patch = np.concatenate((patch,show_patch),axis=0)
            map_p += map_points
    return patch,map_p,sucesses