import torch
import matplotlib.pyplot as plt
import copy
import cv2
import numpy as np

def postprocess(pred):
    pred = pred['instances']._fields

    pred_boxes = pred['pred_boxes']
    scores = pred['scores']
    pred_classes = pred['pred_classes']

    index = torch.where(scores>0.2)[0]
    pred_boxes = pred_boxes[index]
    pred_classes = pred_classes[index]
    scores = scores[index]

    unk_only = (pred_classes == 20)
    return pred_boxes,pred_classes, unk_only



def plot_openset(img,pred_boxes,pred_classes,CLASS_NAMES):
    # print(cos_sim)
    plt.figure(figsize=(15,15))
    demo_image = copy.deepcopy(img)
    for bbox,label in zip(pred_boxes,pred_classes):
        if label == 20:
            color = (0,255,0)
        else:
            color = (255,0,0)
        # color = (255,0,0)
        cv2.rectangle(demo_image, (int(bbox[0]), int(bbox[1])), 
                            (int(bbox[2]),int(bbox[3])), color, 2)
        cv2.putText(demo_image, CLASS_NAMES[int(label)], 
                                (int(bbox[0]), int(bbox[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # cv2.putText(demo_image, str(sc.item()), 
        #                         (int(bbox[0]), int(bbox[1])),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    demo_image = cv2.cvtColor(demo_image, cv2.COLOR_RGB2BGR)
    plt.title("Predicted - MLN unct")
    plt.imshow(demo_image)
    plt.axis('off')
    plt.show()

def plot_candidate(show_patches,show_points,new_query_object_name,scenemap,store=False,scene_name=None,file_path=None):
    size = len(show_patches)
    plt.figure(figsize=(3*size,5))
    plt.suptitle("Candidate Image of [{}]".format(new_query_object_name))
    if len(show_patches)<70:
        for e, (patch,point) in enumerate(zip(show_patches,show_points)):
            map = scenemap.grid_map.copy()
            new_grid = scenemap.xyz2grid(point)
            try:
                map[new_grid[0],new_grid[1],:] = [1,0.5,1]
            except:
                pass
            plt.subplot(2,size,e+1)
            plt.imshow(patch)
            plt.axis('off')

            plt.subplot(2,size,e+size+1)
            map = np.rot90(map)
            plt.imshow(map)
            plt.axis('off')
        if store:
            plt.savefig("{}/{}_{}.png".format(file_path,scene_name,new_query_object_name))
        else:
            plt.show()