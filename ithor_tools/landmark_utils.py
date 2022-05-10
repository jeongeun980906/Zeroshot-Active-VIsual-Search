import copy
import matplotlib.pyplot as plt
import numpy as np
import cv2

landmark_names = ['Bed', 'DiningTable', 'StoveBurner', 'Toilet', 'Sink', 'Desk',
                        'CounterTop','Television','Sofa','SideTable','CoffeeTable','ShelvingUnit','ArmChair']
Word_Dict = {
    'Bed': 'bed', 'DiningTable': 'dining table', 'StoveBurner': 'stove', 'Toilet': 'toilet', 'Sink': 'sink',
    'Desk': 'Desk', 'CounterTop':'kitchen table', 'Sofa':'sofa', 'Television':'television', 'SideTable':'table', 
        'CoffeeTable':'round table','ShelvingUnit':'shelving','ArmChair':'arm chair'}
def choose_ladmark(objects):
    landmarks = []
    for obj in objects:
        if obj['objectType'] in landmark_names:
            cp = obj["position"]
            flag = True
            for l in landmarks:
                if abs(l['cp']['x']-cp['x'])+ abs(l['cp']['z']-cp['z']) < 0.7: #and l['name']==obj['objectType']
                    flag = False
                    break
            if flag:
                landmarks.append(dict(cp = cp, name=obj['objectType']))
            
    visible_landmark_name = []
    for l in landmarks:
        if l['name'] not in visible_landmark_name:
            visible_landmark_name.append(l['name'])
    return landmarks,visible_landmark_name

def gather(controller,query_object,step = 4):
    frames = []
    frames.append(controller.last_event.cv2img)
    pos = [dict(pos = controller.last_event.metadata['agent']['position'], 
                    rot = controller.last_event.metadata['agent']['rotation'])]
    vis = check_visbility(controller.last_event,query_object)
    for _ in range(step-1):
        controller.step(action = "RotateRight", degrees = 360/step)
        pos.append(dict(pos = controller.last_event.metadata['agent']['position'], 
                    rot = controller.last_event.metadata['agent']['rotation']))
        frames.append(controller.last_event.cv2img)
        vis += check_visbility(controller.last_event,query_object)
    return frames,pos,vis

def check_visbility(event,query_object):
    objs = event.metadata['objects']
    for obj in objs:
        if obj['objectType'] == query_object and obj['visible']:
            return True
    return False

def vis_panorama(frames):
    plt.figure(figsize=(10,5))
    col = len(frames)
    angle = 360/col
    for e, frame in enumerate(frames):
        plt.subplot(1,col,e+1)
        plt.title("{}".format(e*angle))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        plt.imshow(frame)
        plt.axis('off')
    plt.show()