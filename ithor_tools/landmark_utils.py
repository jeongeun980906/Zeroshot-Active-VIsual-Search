import copy
from itsdangerous import exc
import matplotlib.pyplot as plt
import numpy as np
import cv2

landmark_names = ['Bed', 'DiningTable', 'StoveBurner', 'Toilet', 'Sink', 'Desk','Drawer','Shelf',
                        'CounterTop','Television','Sofa','SideTable','CoffeeTable','ShelvingUnit','ArmChair','TVStand']
Word_Dict = {
    'Bed': 'bed', 'DiningTable': 'dining table', 'StoveBurner': 'stove', 'Toilet': 'toilet', 'Sink': 'sink',
    'Desk': 'Desk', 'CounterTop':'kitchen table', 'Sofa':'sofa', 'Television':'television','Drawer':'drawer',
     'SideTable':'side table', 'CoffeeTable':'coffee table','ShelvingUnit':'shelving unit','ArmChair':'arm chair','TVStand':'tv stand',
     'Shelf':'shelf'}

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

def gather(controller,query_object,step = 4,angle = 180, show=False):
    frames = []
    gt_boxes = []
    frames.append(controller.last_event.cv2img)
    pos = [dict(pos = controller.last_event.metadata['agent']['position'], 
                    rot = controller.last_event.metadata['agent']['rotation'])]
    vis = check_visbility(controller.last_event,query_object)
    gt_boxes.append(get_gt_box(controller,query_object))
    controller.step("LookDown")
    pos.append(dict(pos = controller.last_event.metadata['agent']['position'], 
                rot = controller.last_event.metadata['agent']['rotation']))
    frames.append(controller.last_event.cv2img)
    vis += check_visbility(controller.last_event,query_object)
    gt_boxes.append(get_gt_box(controller,query_object))
    controller.step("LookUp")
    for _ in range(step-1):
        controller.step(action = "RotateRight", degrees = angle/(step-1))
        pos.append(dict(pos = controller.last_event.metadata['agent']['position'], 
                    rot = controller.last_event.metadata['agent']['rotation']))
        frames.append(controller.last_event.cv2img)
        vis += check_visbility(controller.last_event,query_object)
        gt_boxes.append(get_gt_box(controller,query_object))
        controller.step("LookDown")
        pos.append(dict(pos = controller.last_event.metadata['agent']['position'], 
                    rot = controller.last_event.metadata['agent']['rotation']))
        frames.append(controller.last_event.cv2img)
        vis += check_visbility(controller.last_event,query_object)
        gt_boxes.append(get_gt_box(controller,query_object))
        controller.step("LookUp")
    return frames,pos,gt_boxes,vis

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

def get_gt_box(controller,query_object_IDs,show=False):
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
    try:
        GT_box = [min(temp[1]),min(temp[0]),max(temp[1]),max(temp[0])]
        return GT_box
    except:
        return None
    # plt.imshow(instance_segmentation)
    # plt.axis('off')
    # plt.show()
    