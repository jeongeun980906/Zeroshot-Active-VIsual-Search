# importing sys
import sys
import random
import argparse

from zmq import device
# adding Folder_2 to the system path
sys.path.insert(0, '/home/jeongeun/faster_rcnn_rilab')
import numpy as np
import torch

# AI2 THOR
import ai2thor
from ai2thor.controller import Controller,BFSController
from ai2thor.platform import CloudRendering
from eval_ithor.reset import reset_scene,load_detector,get_min_dis,move_init,load_detector_base
from eval_ithor.objects import choose_query_objects,detect
from eval_ithor.score import score_storage
from ithor_tools.map2 import single_scenemap
from ithor_tools.landmark_utils import gather,vis_panorama,Word_Dict,choose_ladmark
from ithor_tools.transform import cornerpoint_projection,depth2world
from ithor_tools.vis_tool import vis_visit_landmark

# Co occurance module
from co_occurance.comet_co import co_occurance_score,co_occurance_score_base
from co_occurance.move import co_occurance_based_schedular

from detector.postprocess import postprocess,plot_openset,plot_candidate
# Matching Module
from detector.query_matching import matcher

# Planning Module
from RRT import gridmaprrt as rrt
from RRT import gridmaprrt_pathsmoothing as smoothing

kitchens = [f"FloorPlan{i}_physics" for i in range(1, 31)]
living_rooms = [f"FloorPlan{200 + i}_physics" for i in range(1, 31)]
bedrooms = [f"FloorPlan{300 + i}_physics" for i in range(1, 31)]
bathrooms = [f"FloorPlan{400 + i}_physics" for i in range(1, 31)]
train = [f"FloorPlan_Train{i}_{j}" for i in range(1,12) for j in range(1,5)]

def get_scene(args):
    if args.scene == 'all':
        return train
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
    co_thres = args.co_thres
    angle = 60
    step = 3
    # random.seed(10)
    ST = score_storage()

    # random.seed()
    device = 'cuda:{}'.format(args.gpu)
    scene_names = get_scene(args)
    scene_names = random.choices(train,k=10)
    print(scene_names)
    # scene_names = train
    '''
    Load co occurance measure
    '''
    if args.co_base:
        co_occurance_scoring = co_occurance_score_base()
    else:
        co_occurance_scoring = co_occurance_score(device)
    '''
    Load Detector
    '''
    if args.base_detector:
        predictor = load_detector_base(device=device)
    else:
        predictor = load_detector(device=device,ID=args.detector_id)

    for scene_name in scene_names:
        print(scene_name)
        gridSize=0.05

        controller = Controller(
            platform = CloudRendering,
            agentMode="locobot",
            visibilityDistance=5.0,
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

        objects,rstate = reset_scene(controller)

        landmarks,visible_landmark_name = choose_ladmark(objects)
        landmark_cat = [Word_Dict[l] for l in visible_landmark_name]

        co_occurance_scoring.landmark_init(landmark_cat)

        query_objects = choose_query_objects(objects,'all')

        scene_bounds = controller.last_event.metadata['sceneBounds']['cornerPoints']
        scene_bounds = cornerpoint_projection(scene_bounds)

        sm = single_scenemap(scene_bounds,rstate,stepsize = 0.1,
                        landmark_names=visible_landmark_name,landmarks=landmarks,num_loi = args.num_loi)
        sm.plot_landmarks(controller)

        landmark_config = dict(name=visible_landmark_name,color = sm.landmark_colors)
        d2w = depth2world()
        """"
        Load Planner
        """
        rrtplanner = rrt.RRT(controller = controller, expand_dis=0.1,max_iter=10000,goal_sample_rate=20)
        print(len(query_objects))
        for query_object in query_objects:
            move_init(controller,rstate)
            query_object_name = query_object['objectType']
            print(query_object_name)
            # imshow_grid = sm.plot(controller.last_event.metadata['agent']['position'],query_object['position'])
            # plot_frames(controller.last_event,imshow_grid,landmark_config)
            '''
            Load Matcher
            '''
            query_matcher = matcher(query_object_name,threshold=args.clip_thres,device = device)
            """
            co occurance score
            """
            res = co_occurance_scoring.score(query_object_name)
            if max(res)<co_thres:
                co_thres = max(res)-0.2
            print(res,visible_landmark_name)

            move_init(controller,rstate)
            schedular = co_occurance_based_schedular(landmarks,visible_landmark_name)
            schedular.get_node(sm,controller,res,co_thres)
            move_init(controller,rstate)
            schedular.get_edge(controller)
            path = schedular.optimize()
            vis_visit_landmark(query_object,path,controller,sm,landmark_config,store=True)
            
            move_init(controller,rstate)
            min_dis = get_min_dis(query_object,controller,sm,schedular)
            total_patch = np.zeros((0,256,256,3),dtype=np.uint8)
            total_mappoints = []
            total_success = 0
            total_path_len = 0
            for e,p in enumerate(path[1:]):
                if e > 15:
                    break
                pos = controller.last_event.metadata['agent']['position']
                rrtplanner.set_start(pos)
                rrtplanner.set_goal(p[0])
                print("start planning")
                try:
                    local_path = rrtplanner.planning(animation=False)
                except:
                    break
                try:
                    smoothpath = smoothing.path_smoothing(rrtplanner,40,verbose=False)
                except:
                    smoothpath = local_path
                print("end planning")
                rrtplanner.plot_path(smoothpath)
                
                flag,path_len,frames = rrtplanner.go_with_teleport(smoothpath,maxspeed=0.2)
                total_path_len += path_len
                # video = ImageSequenceClip(frames, fps=10)
                # video.write_gif('temp.gif')
                # with open('temp.gif','rb') as file:
                #     display(IM(file.read(),width = 300))
                
                pos = controller.last_event.metadata['agent']['position']
                controller.step(
                    action="Teleport",
                    position = pos, rotation = dict(x=0,y=p[1]-angle/2,z=0)
                        )
                print("end move")
                # imshow_grid = sm.plot(pos,query_object['position'])
                # plot_frames(controller.last_event,imshow_grid,landmark_config)
                frames, single_pos,gt_boxes,gt_vis = gather(controller,[query_object['objectId']],step=step,angle=angle)
                print('gt_vis?',gt_vis)
                candidate_patches, candidate_map_points,sucesses = detect(frames,single_pos,gt_boxes,controller,predictor,query_matcher,d2w)
                total_patch = np.concatenate((total_patch,candidate_patches),axis=0)
                total_mappoints += candidate_map_points
                total_success += sucesses
                # vis_panorama(frames,res=angle)
                print(len(total_patch))
                if gt_vis:
                    # total_patch = np.concatenate(total_patch,axis=0)
                    # print(total_patch.shape,)
                    break
                if len(total_patch)>10 and p[-1]<0.9:
                    break
            print("Sucess?",total_success>0)
            print("Total Path Length", total_path_len)
            if len(total_patch)>0:
                plot_candidate(total_patch,total_mappoints,query_object_name,sm,store=True,scene_name=scene_name)
            SPL = (total_success>0)*min_dis/(total_path_len+1e-6)
            ST.append(SPL,query_object_name,scene_name)
        controller.stop()
    df = ST.average()
    print(df)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu',default = 1, type=int,help='gpu setup')
    parser.add_argument('--scene',default='all',choices=['all','bed','kitchen','living_room','bath'],help='type of scene')
    
    # Dector setup
    parser.add_argument('--base_detector',action='store_true' ,default=False,help='Use baseline Detector')
    parser.add_argument('--detector_id',type=int,default=19,help='detector id')

    # Co occurance measure setup
    parser.add_argument('--co_base',action='store_true' ,default=False,help='co occurance as word2vec')
    parser.add_argument('--co_thres',default = 0.6, type=float,help='co occurance threshold')

    # CLIP setup
    parser.add_argument('--clip_thres',default = 29, type=float,help='clip threshold')

    # LOI setup
    parser.add_argument('--num_loi',default = 2, type=int,help='number of lois: 1,2 or 0 (all)')
    
    args = parser.parse_args()

    main(args)