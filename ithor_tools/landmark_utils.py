import copy
from itsdangerous import exc
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from detector.postprocess import plot_openset
import math

# landmark_names = ['Bed', 'DiningTable', 'StoveBurner', 'Toilet', 'Sink', 'Desk','Drawer','Shelf',
#                         'CounterTop','Television','Sofa','SideTable','CoffeeTable','ShelvingUnit','ArmChair','TVStand']
# Word_Dict = {
#     'Bed': 'bed', 'DiningTable': 'dining table', 'StoveBurner': 'stove', 'Toilet': 'toilet', 'Sink': 'sink',
#     'Desk': 'Desk', 'CounterTop':'kitchen table', 'Sofa':'sofa', 'Television':'television','Drawer':'drawer',
#      'SideTable':'side table', 'CoffeeTable':'coffee table','ShelvingUnit':'shelving unit','ArmChair':'arm chair','TVStand':'tv stand',
#      'Shelf':'shelf'}

in_landmark_names = ['diningtable','sofa','tvmonitor']
out_landmark_names = ['desk','drawer','side table','coffee table','bed','arm chair']
landmark_names = in_landmark_names+out_landmark_names

def choose_ladmark(objects):
    landmarks = []
    temp_ln = copy.deepcopy(landmark_names)
    temp_ln.remove('Shelf')
    temp_ln.remove("Drawer")
    temp_ln.remove("TVStand")
    for obj in objects:
        if obj['objectType'] in temp_ln:
            cp = obj["position"]
            flag = True
            for l in landmarks:
                if abs(l['cp']['x']-cp['x'])+ abs(l['cp']['z']-cp['z']) < 0.7 and l['name']==obj['objectType']: 
                    l['ID'].append(obj['objectId'])
                    flag = False
                    break
            if flag:
                landmarks.append(dict(cp = cp, name=obj['objectType'],ID = [obj['objectId']]))
            
    for obj in objects:
        if obj['objectType'] == 'Shelf' or obj['objectType'] =='Drawer' or obj['objectType'] == 'TVStand':
            cp = obj["position"]
            flag = True
            for l in landmarks:
                if abs(l['cp']['x']-cp['x'])+ abs(l['cp']['z']-cp['z']) < 0.7:
                    if obj['objectType'] == 'TVStand':
                        if l['name'] == 'DiningTable': 
                            l['ID'].append(obj['objectId'])
                    else:
                        l['ID'].append(obj['objectId'])
                    flag = False
            if flag:
                landmarks.append(dict(cp = cp, name=obj['objectType'],ID = [obj['objectId']]))
    visible_landmark_name = []
    for l in landmarks:
        if l['name'] not in visible_landmark_name:
            visible_landmark_name.append(l['name'])
    return landmarks,visible_landmark_name

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
                    detection_labels,scene_memory,unk_flag=True):
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
        # plot_openset(img,pred_boxes,pred_classes,landmark_names)
        # l = m.detect_gt_box(controller)
        l = scene_memory.proj(pred_boxes,pred_classes,pred_uncts,controller,landmark_names)
        if l != None:
            detected_landmarks += l
    detected_landmarks = list(set(detected_landmarks))
    return frames,pos,gt_boxes,vis,detected_landmarks


def check_visbility(event,query_object):
    objs = event.metadata['objects']
    for obj in objs:
        if obj['objectId'] == query_object[0] and obj['visible']:
            return True
    return False

def vis_panorama(frames,res=360):
    plt.figure(figsize=(10,5))
    col = len(frames)
    angle = res/col
    for e, frame in enumerate(frames):
        plt.subplot(1,col,e+1)
        plt.title("{}".format(e))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        plt.imshow(frame)
        plt.axis('off')
    plt.show()

def get_gt_box(controller,query_object_IDs,show=False,version=1,opos=None):
    instance_segmentation = controller.last_event.instance_segmentation_frame
    obj_colors = controller.last_event.object_id_to_color
    temp = np.zeros((instance_segmentation.shape[0],instance_segmentation.shape[1]))    
    for query_object_ID in query_object_IDs:
        query_color = obj_colors[query_object_ID]
        
        # print(controller.last_event.object_id_to_color)
        R = (instance_segmentation[:,:,0]==query_color[0])
        G = (instance_segmentation[:,:,1]==query_color[1])
        B = (instance_segmentation[:,:,2]==query_color[2])
        total = R & G & B
        temp[total] = +1
    if show:
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(controller.last_event.frame)
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.imshow(temp)
        plt.axis('off')
        plt.show()

    # thres = np.max(temp)
    # if thres < 3:
    #     thres = 3
    temp = np.where(temp>=1)
    cpos = controller.last_event.metadata['agent']['position']
    dis = math.sqrt((cpos['x']-opos['x'])**2+(cpos['z']-opos['z'])**2)    
    try:
        GT_box = [min(temp[1]),min(temp[0]),max(temp[1]),max(temp[0])]
        area = (GT_box[2]-GT_box[0])*(GT_box[3]-GT_box[1])
        # print(area/((instance_segmentation.shape[0]*instance_segmentation.shape[1])))
        thres = 1e-3 if version==1 else 1e-3
        # print(dis)
        if area>thres*(instance_segmentation.shape[0]*instance_segmentation.shape[1]) and dis<2.5:#2.5:
            return GT_box
        else:
            return None
    except:
        return None
    # plt.imshow(instance_segmentation)
    # plt.axis('off')
    # plt.show()

