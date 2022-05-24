import numpy as np
import torch
from detector.postprocess import postprocess

object_bed = [
    'AlarmClock','Book', "CellPhone",'CreditCard',"KeyChain", "Mug", "Pillow","CD","Laptop",
    "TeddyBear","TissueBox"]
    
object_kitchen = ['Book', 'Apple', 'Bread', "DishSponge" ,"Kettle","Pot","PepperShaker",
                "Mug","Pan","Toaster","PaperTowelRoll"]

object_living_room = ['Book',"CellPhone",'CreditCard',"KeyChain", "RemoteControl","TissueBox","Watch","Laptop"]

object_bath = ['HandTowel',"SoapBar","SprayBottle","TissueBox","ToiletPaper","Towel"]

total = list(set(object_bed+object_kitchen+object_living_room+object_bath))

def get_obj_list(scene_type):
    if scene_type == 'all':
        return total
    elif scene_type == 'bed':
        return object_bed
    elif scene_type == 'kitchen':
        return object_kitchen
    elif scene_type == 'living_room':
        return object_living_room
    elif scene_type == 'bath':
        return object_bath
    else:
        raise NotImplementedError

def choose_query_objects(objects,scene_type='all'):
    object_list= get_obj_list(scene_type)
    query_objects = []
    for obj in objects:
        if obj['objectType'] in object_list:
            query_objects.append(obj)
    return query_objects


def detect(frames,single_pos,gt_boxes,controller,predictor,matcher,d2w,unk_only_flag=True):
    patch = np.zeros((0,256,256,3),dtype=np.uint8)
    map_p = []
    sucesses = 0
    for frame,pos,gt_box in zip(frames,single_pos,gt_boxes):
        pred = predictor(frame)
        pred_boxes, pred_classes,unk_only = postprocess(pred)
        # plot_openset(frame,pred_boxes,pred_classes,VOC_CLASS_NAMES)
        if unk_only_flag:
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