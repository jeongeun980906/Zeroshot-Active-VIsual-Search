import numpy as np
import matplotlib.pyplot as plt
import math

def check_visbility(event,query_object):
    objs = event.metadata['objects']
    for obj in objs:
        if obj['objectId'] == query_object[0] and obj['visible']:
            return True
    return False


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