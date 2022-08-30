import numpy as np
import torch


total = ['RemoteControl','Laptop','Book','Apple','CD',
            'Pot','Bowl','AlarmClock','TeddyBear',
            'CellPhone','SprayBottle','Pillow']

object_bed = [
    'AlarmClock','Book', "CellPhone", "Pillow","CD","Laptop",
    "TeddyBear"]
    
object_kitchen = ['Apple',"Pot", "Bowl", "Pan","Toaster"]

object_living_room = ['Book',"CellPhone", "RemoteControl","Laptop"]

object_bath = ["SoapBar","SprayBottle","TissueBox",
                    "ToiletPaper","Towel"]

# total = list(set(object_bed+object_kitchen+object_living_room+object_bath))

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


