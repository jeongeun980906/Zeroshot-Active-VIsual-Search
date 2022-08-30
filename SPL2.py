# importing sys
import sys
import random
import argparse
import math

from zmq import device
# adding Folder_2 to the system path
try:
    # adding Folder_2 to the system path
    sys.path.insert(0, '/home/jeongeun/test_env/Open-Set-Object-Detection')
    from data.phase_1 import load_voc_instances,VOC_CLASS_NAMES
except:
    print("Import Error")
    raise ImportError

import numpy as np
import torch
from tqdm import tqdm

# AI2 THOR
import ai2thor
from ai2thor.controller import Controller,BFSController
from ai2thor.platform import CloudRendering
from eval_ithor.reset import reset_scene,load_detector,get_min_dis,move_init,load_detector_base
from eval_ithor.objects import choose_query_objects
from eval_ithor.score import score_storage

from detector.thor import gather,gather3,detect
from ithor_tools.landmark_utils import landmark_names,in_landmark_names,out_landmark_names
from ithor_tools.transform import cornerpoint_projection,depth2world
from ithor_tools.vis_tool import vis_visit_landmark,draw_path,vis_panorama

# Co occurance module
from FBE.memory import fbe_map
from FBE.schedular import traj_schedular
from co_occurance.comet_co import co_occurance_score,co_occurance_score_base
from co_occurance.fliker import fliker_knowledge

from detector.postprocess import postprocess,plot_openset,plot_candidate
# Matching Module
from detector.query_matching import matcher


# Planning Module
from RRT import gridmaprrt as rrt
from RRT import gridmaprrt_pathsmoothing as smoothing

kitchens = [f"FloorPlan{i}_physics" for i in range(21, 31)]
living_rooms = [f"FloorPlan{200 + i}_physics" for i in range(21, 31)]
bedrooms = [f"FloorPlan{300 + i}_physics" for i in range(21, 31)]
bathrooms = [f"FloorPlan{400 + i}_physics" for i in range(21, 31)]
train = [f"FloorPlan_Train{i}_{j}" for i in range(1,12) for j in range(1,6)]
val = [f"FloorPlan_Val{i}_{j}" for i in range(1,4) for j in range(1,6)]

object_names = ['RemoteControl','Laptop','Book','Apple','CD',
            'Pot','Bowl','AlarmClock','TeddyBear',
            'CellPhone','SprayBottle','Pillow']
# visible_landmark_names = ['Bed', 'DiningTable', 'StoveBurner', 'Desk','Drawer','Television','Sofa',
#                 'SideTable','CoffeeTable','ArmChair',]

detection_labels = []
for l in in_landmark_names:
    detection_labels.append(VOC_CLASS_NAMES.index(l.replace(" ","")))

def get_scene(args):
    if args.scene == 'all':
        return val #train
    elif args.scene == 'bed':
        return bedrooms
    elif args.scene == 'living_room':
        return living_rooms
    elif args.scene == 'bath':
        return bathrooms
    elif args.scene == 'kitchen':
        return kitchens
    else:
        raise NotImplementedError
    
def main(args):
    ROBOT_NAME = 'locobot' if args.scene == 'all' else 'default'
    angle = 200
    step = 5
    # random.seed(10)
    ST = score_storage(args)
    if args.resume:
        ST.load_json()

    # random.seed()
    device = 'cuda:{}'.format(args.gpu)
    scene_names = get_scene(args)
    scene_names = scene_names[12:]
    # scene_names = [scene_names[6]]
    # scene_names = random.choices(train,k=10)
    # scene_names = ['FloorPlan_Train8_1']
    # scene_names =  ['FloorPlan_Train1_2', 'FloorPlan_Train10_4','FloorPlan_Train9_1','FloorPlan_Train6_2',
    #             'FloorPlan_Train2_3','FloorPlan_Train5_1']#, 'FloorPlan_Train8_1'] #'
    #            ,'FloorPlan_Train8_2','FloorPlan_Train4_1','FloorPlan_Train5_3', 'FloorPlan_Train2_2','FloorPlan_Train3_2']
    print(scene_names)
    # scene_names = ['FloorPlan_Train1_2']
    # scene_names = train
    ''''
    Load Co-occuernce
    '''
    # landmark_cat = [Word_Dict[l] for l in landmark_names]
    if not args.dis_only:
        if args.word_dis:
            co_occurance_scoring = co_occurance_score_base()
        elif args.fliker:
            co_occurance_scoring = fliker_knowledge()
        else:
            co_occurance_scoring = co_occurance_score(device)
        co_occurance_scoring.landmark_init(landmark_names)
    
    '''
    Load Detector
    '''
    if args.base_detector:
        predictor = load_detector_base(device=device)
        unk_only_flag = False
    else:
        predictor = load_detector(device=device,ID=args.detector_id)
        unk_only_flag = True
    
    clip_matcher = matcher(out_landmark_names,device=device)

    for scene_name in scene_names:
        print(scene_name)
        gridSize=0.05
        
        controller = Controller(
            platform = CloudRendering,
            agentMode= ROBOT_NAME,
            visibilityDistance=2.5, # 1.5
            scene = scene_name,
            gridSize=gridSize,
            movementGaussianSigma=0,
            rotateStepDegrees=90,
            rotateGaussianSigma=0,
            renderClassImage = False,
            renderDepthImage=False,
            renderInstanceSegmentation=True,
            width=300,
            height=300,
            fieldOfView=60
        )

        objects,rstate = reset_scene(controller,args)
        scene_bounds = controller.last_event.metadata['sceneBounds']['cornerPoints']
        scene_bounds = cornerpoint_projection(scene_bounds)
        
        
        sche = traj_schedular(landmark_names,controller,args=args)

        query_objects = choose_query_objects(objects,args.scene)

        
        move_init(controller,rstate)
        d2w = depth2world()
        """"
        Load Planner
        """
        rrtplanner = rrt.RRT(controller = controller, expand_dis=0.1,max_iter=10000,goal_sample_rate=20)
        # print(len(query_objects))
        for query_object in tqdm(query_objects,desc=scene_name):
            random.seed(100)
            np.random.seed(100)
            torch.random.manual_seed(100)
            
            NUM_WAYPOINT = 0
            scene_memory = fbe_map(scene_bounds,rstate,landmark_names=landmark_names, stepsize=0.1)
            move_init(controller,rstate)
            traj_image = controller.last_event.third_party_camera_frames[0]
            scene_memory.reset_gridmap()
            opos = query_object['position']
            co_thres = args.co_thres
            if not args.dis_only:
                co_occurance = co_occurance_scoring.score(query_object['objectType'])
                # if max(co_occurance)<co_thres:
                #     co_thres = max(co_thres)-0.2
            else:
                co_occurance = [0]*len(landmark_names)
            
            sche.set_score(co_occurance)
            clip_matcher.tokenize(query_object['objectType'])

            min_dis = get_min_dis(query_object,controller,None,None)
            if min_dis == None:
                continue
            total_patch = np.zeros((0,256,256,3),dtype=np.uint8)
            total_mappoints = []
            total_success = 0
            total_path_len = 0
            '''
            search start
            '''
            while not total_success:
                '''
                inintial scanning
                '''
                frames,single_pos,gt_boxes,gt_vis,detected_landmarks = gather3(controller,[query_object['objectId']],opos,
                    predictor,postprocess,clip_matcher,
                    detection_labels,scene_memory)
                # print('gt_vis?',gt_vis)
                candidate_patches, candidate_map_points,sucesses = detect(frames,single_pos,gt_boxes,controller,predictor,clip_matcher,d2w,unk_only_flag = unk_only_flag)

                total_patch = np.concatenate((total_patch,candidate_patches),axis=0)
                total_mappoints += candidate_map_points
                total_success += sucesses
            
                scene_memory.visited(controller)
                # landmark_config = dict(name=landmark_names,color = scene_memory.landmark_colors)
                cpos = controller.last_event.metadata['agent']['position']
                candidate_trajs = []
                uncts = []
                for l in detected_landmarks:
                    res = scene_memory.get_reachable(cpos,l)
                    if not res[0] == None:
                        uncts.append(res[3])
                        res = dict(name=l,pos=res[0],rot = res[1])
                        candidate_trajs.append(res)
                        
                waypoints = scene_memory.frontier_detection(cpos)
                candidate_trajs += waypoints
                uncts += [0]*len(waypoints)
                flag = 0
                cpos = controller.last_event.metadata['agent']['position']
                '''
                waypoint generation
                '''
                path = sche.schedule(cpos,0,candidate_trajs,uncts,co_thres = co_thres)
                # print(path)
                for p in path[1:]:
                    if p == None:
                        print("Done exploration")
                        flag +=1
                        scene_memory.reset_gridmap()
                        break
                    # print(p)
                    flag = 0
                    pos = controller.last_event.metadata['agent']['position']
                    rrtplanner.set_start(pos)
                    rrtplanner.set_goal(p['pos'])
                    local_path = rrtplanner.planning(animation=False)
                    try:
                        smoothpath = smoothing.path_smoothing(rrtplanner,40,verbose=False)
                    except:
                        smoothpath = local_path
                    # print(smoothpath,pos,p['pos'])
                    path_len = 0
    
                    if smoothpath != None:
                        fr_pos = rrtplanner.rstate[smoothpath[0]]
                        for idx in smoothpath[1:]:
                            to_pos = rrtplanner.rstate[idx]
                            # Get constant distanced points
                            delta = to_pos - fr_pos
                            # theta = math.atan2(delta[1],delta[0])
                            d = np.linalg.norm(delta)
                            path_len += d
                            fr_pos = to_pos
                    # print(path_len)
                    NUM_WAYPOINT += 1
                    traj_image = draw_path(rrtplanner,traj_image,smoothpath,NUM_WAYPOINT = NUM_WAYPOINT,scene_name=scene_name,file_path=ST.file_path,query_name=query_object['objectType'])
                    total_path_len += path_len
                    goal_pos = dict(x=to_pos[0],y=rstate[0]['y'],z=to_pos[1])
                    controller.step(
                    action="Teleport",
                    position = goal_pos, rotation = dict(x=0,y=0,z=0)
                        )
                    if not controller.last_event.metadata['lastActionSuccess']:
                            print("path failed")
                    # print(controller.last_event.metadata['lastActionSuccess'])
                    # 
                    # to_pos = rrtplanner.rstate[smoothpath[-1]]
                    # rrtplanner.plot_path(smoothpath)
                    # flag,path_len,frames = rrtplanner.go_with_teleport(smoothpath,maxspeed=0.2)
                    # total_path_len += path_len

                    if p['name'] == 'frontier':
                        '''
                        if the robot reaches the frontier
                        '''
                        scene_memory.reset_landmark(detected_landmarks)
                        break
                    else:
                        pos = controller.last_event.metadata['agent']['position']
                        controller.step(
                            action="Teleport",
                            position = pos, rotation = dict(x=0,y=p['rot']-angle/2,z=0)
                                )
                        if not controller.last_event.metadata['lastActionSuccess']:
                            print("path failed")
                            continue
                        frames, single_pos,gt_boxes,gt_vis = gather(controller,[query_object['objectId']],opos,step=step,angle=angle)
                        candidate_patches, candidate_map_points,sucesses = detect(frames,single_pos,gt_boxes,controller,predictor,clip_matcher,d2w,unk_only_flag = unk_only_flag)
                        total_patch = np.concatenate((total_patch,candidate_patches),axis=0)
                        total_mappoints += candidate_map_points
                        total_success += sucesses
                    # print(total_path_len)
                    if total_success or total_path_len>50:
                        break
                if total_success or total_path_len>50 or flag>4:
                        break
            if len(total_patch)>0:
                plot_candidate(total_patch,total_mappoints,query_object['objectType'],scene_memory,store=True,scene_name=scene_name,file_path=ST.file_path)
            if total_path_len == 0:
                total_path_len += 0.01
            SPL = (total_success>0)*min_dis/(total_path_len)
            # print("SPL:", SPL)
            ST.append(SPL,query_object['objectType'],scene_name,NUM_WAYPOINT)
            ST.save_json()
            del scene_memory
            del total_patch
        # del 
        controller.stop()
    df = ST.average()
    print(df)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version',default = 3, type=int,help='version')
    parser.add_argument('--resume',action='store_true' ,default=False,help='Resume from json file')
    parser.add_argument('--val',action='store_true' ,default=False,help='val set')
    parser.add_argument('--gpu',default = 1, type=int,help='gpu setup')
    parser.add_argument('--scene',default='all',choices=['all','bed','kitchen','living_room','bath'],help='type of scene')
    
    # Dector setup
    parser.add_argument('--base_detector',action='store_true' ,default=False,help='Use baseline Detector')
    parser.add_argument('--detector_id',type=int,default=19,help='detector id')

    # Co occurance measure setup
    parser.add_argument('--dis_only',action='store_true' ,default=False,help='no co-occurance')
    parser.add_argument('--fliker',action='store_true' ,default=False,help='co-occurance fliker')
    parser.add_argument('--word_dis',action='store_true' ,default=False,help='co-occurance word_distance')
    parser.add_argument('--co_thres',type = float ,default=0.2,help='co-occurance threshold')
    parser.add_argument('--unct_ratio',type = float ,default=0.2,help='unct ratio')
    # CLIP setup
    parser.add_argument('--clip_thres',default = 29, type=float,help='clip threshold')
    args = parser.parse_args()

    main(args)