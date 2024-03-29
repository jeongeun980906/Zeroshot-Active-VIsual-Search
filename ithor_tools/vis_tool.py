from IPython.display import HTML, display
import sys
import imageio
from moviepy.editor import ImageSequenceClip
from typing import Sequence
import numpy as np
import os
from typing import Optional
import ai2thor.server
from typing import Union
import seaborn as sns
import cv2
import copy
import matplotlib.pyplot as plt

def plot_frames(event: Union[ai2thor.server.Event, np.ndarray],gridmap : np.ndarray, landmark_config) -> None:
    """Visualize all the frames on an AI2-THOR Event.
    Example:
    plot_frames(controller.last_event)
    """
    if isinstance(event, ai2thor.server.Event):
        third_person_frames = event.third_party_camera_frames
        RGB = event.frame
        DEPTH = event.depth_frame

        # Set up the axes with gridspec
        fig = plt.figure(figsize=(7, 14))
        grid = plt.GridSpec(13, 6, wspace=0.4, hspace=0.3)

        ax = fig.add_subplot(grid[:4, :3])
        im = ax.imshow(RGB)
        ax.axis("off")
        ax.set_title('RGB')

        ax = fig.add_subplot(grid[:4, 3:])
        im = ax.imshow(DEPTH)
        ax.axis("off")
        ax.set_title('DEPTH')
        fig.colorbar(im, fraction=0.046, pad=0.04, ax=ax)

        # add third party camera frames
        ax = fig.add_subplot(grid[4:8, :])
        ax.set_title("Map View")
        temp = crop_zeros(third_person_frames[0])
        print(temp.shape)
        ax.imshow(temp)
        ax.axis("off")

        ax = fig.add_subplot(grid[8:12, :])
        ax.set_title("Grid Map")
        ax.imshow(gridmap,cmap=plt.cm.gray_r)
        ax.axis("off")
        

        ax = fig.add_subplot(grid[-1, :])
        ax.set_title("Landmark")
        cat_name = landmark_config['name']
        cmap = landmark_config['color']
        img = [i for i in range(len(cat_name))]
        img = np.asarray(img).reshape(1,-1)
        annot = np.asarray(cat_name).reshape(1,-1)
        sns.heatmap(img, cmap=cmap,annot=annot, fmt = '', cbar=False,annot_kws={"size": 9})
        ax.axis('off')

        # plt.tight_layout()
        plt.show()

def show_video(frames: Sequence[np.ndarray], fps: int = 10):
    """Show a video composed of a sequence of frames.
    Example:
    frames = [
        controller.step("RotateRight", degrees=5).frame
        for _ in range(72)
    ]
    show_video(frames, fps=5)
    """
    frames = ImageSequenceClip(frames, fps=fps)
    return frames.ipython_display()



def show_objects_table(objects: list) -> None:
    """Visualizes objects in a way that they are clickable and filterable.
    Example:
    event = controller.step("MoveAhead")
    objects = event.metadata["objects"]
    show_objects_table(objects)
    """
    import pandas as pd
    from collections import OrderedDict

    processed_objects = []
    for obj in objects:
        obj = obj.copy()
        obj["position[x]"] = round(obj["position"]["x"], 4)
        obj["position[y]"] = round(obj["position"]["y"], 4)
        obj["position[z]"] = round(obj["position"]["z"], 4)

        obj["rotation[x]"] = round(obj["rotation"]["x"], 4)
        obj["rotation[y]"] = round(obj["rotation"]["y"], 4)
        obj["rotation[z]"] = round(obj["rotation"]["z"], 4)

        del obj["position"]
        del obj["rotation"]

        # these are too long to display
        del obj["objectOrientedBoundingBox"]
        del obj["axisAlignedBoundingBox"]
        del obj["receptacleObjectIds"]

        obj["mass"] = round(obj["mass"], 4)
        obj["distance"] = round(obj["distance"], 4)

        obj = OrderedDict(obj)
        obj.move_to_end("distance", last=False)
        obj.move_to_end("rotation[z]", last=False)
        obj.move_to_end("rotation[y]", last=False)
        obj.move_to_end("rotation[x]", last=False)

        obj.move_to_end("position[z]", last=False)
        obj.move_to_end("position[y]", last=False)
        obj.move_to_end("position[x]", last=False)

        obj.move_to_end("name", last=False)
        obj.move_to_end("objectId", last=False)
        obj.move_to_end("objectType", last=False)

        processed_objects.append(obj)

    df = pd.DataFrame(processed_objects)
    print(
        "Object Metadata. Not showing objectOrientedBoundingBox, axisAlignedBoundingBox, and receptacleObjectIds for clarity."
    )
    pd.set_option('display.max_rows', None)
    return df


def crop_zeros(image):
    y_nonzero, x_nonzero, _ = np.nonzero(1-image/255)
    return image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]


def vis_visit_landmark(query_object,path,controller,scenemap,landmark_config,store=False):
    temp = copy.deepcopy(scenemap.grid_map)
    temp = np.ascontiguousarray(temp*255, dtype=np.uint8).copy()
    pos = controller.last_event.metadata['agent']['position']
    pos = scenemap.xyz2grid(pos)
    temp[pos[0],pos[1]] = [255,0,0]
    pos = query_object['position']
    pos = scenemap.xyz2grid(pos)
    temp[pos[0],pos[1]] = [0,0,255]
    for points in path[1:]:
        pos = scenemap.xyz2grid(points[0])
        temp[pos[0],pos[1]] = [100,100,200]
    temp = cv2.resize(temp, dsize = (-1,-1),fx = 10,fy=10,interpolation = cv2.INTER_CUBIC)
    h,w = temp.shape[0], temp.shape[1]
    # print(h,w)
    temp = cv2.rotate(temp, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    fig = plt.figure(figsize=(7, 7))
    grid = plt.GridSpec(7, 1, wspace=1, hspace=0.1)

    ax = fig.add_subplot(grid[:6, 0])
    ax.set_title("landmark visit")
    for e,points in enumerate(path[1:]):
        pos = scenemap.xyz2grid(points[0])
        temp = cv2.putText(temp, str(e+1), (10*pos[0]+20, w-10*pos[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),2,cv2.LINE_AA)
        temp = cv2.putText(temp, str(points[2]), (10*pos[0]+60, w-10*pos[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255),2,cv2.LINE_AA)
    plt.axis("off")
    plt.imshow(temp)
    ax = fig.add_subplot(grid[6:, 0])
    ax.set_title("Landmark")
    cat_name = landmark_config['name']
    cmap = landmark_config['color']
    img = [i for i in range(len(cat_name))]
    img = np.asarray(img).reshape(1,-1)
    annot = np.asarray(cat_name).reshape(1,-1)
    sns.heatmap(img, cmap=cmap,annot=annot, fmt = '', cbar=False,annot_kws={"size": 9})
    ax.axis('off')
    if store:
        plt.savefig('./res/visit.png')
    else:
        plt.show()


def plot_frames2(event: Union[ai2thor.server.Event, np.ndarray],gridmap : np.ndarray, landmark_config) -> None:
    """Visualize all the frames on an AI2-THOR Event.
    Example:
    plot_frames(controller.last_event)
    """
    if isinstance(event, ai2thor.server.Event):
        third_person_frames = event.third_party_camera_frames
        RGB = event.frame
        DEPTH = event.depth_frame

        # Set up the axes with gridspec
        fig = plt.figure(figsize=(7, 14))
        grid = plt.GridSpec(16, 6, wspace=0.4, hspace=0.3)

        ax = fig.add_subplot(grid[:5, :3])
        im = ax.imshow(RGB)
        ax.axis("off")
        ax.set_title('RGB')

        ax = fig.add_subplot(grid[:5, 3:6])
        im = ax.imshow(DEPTH)
        ax.axis("off")
        ax.set_title('DEPTH')
        fig.colorbar(im, fraction=0.046, pad=0.04, ax=ax)

        # add third party camera frames
        ax = fig.add_subplot(grid[5:10, :6])
        ax.set_title("Map View")
        temp = crop_zeros(third_person_frames[0])
        ax.imshow(temp)
        ax.axis("off")

        ax = fig.add_subplot(grid[10:15, :6])
        ax.set_title("Memory Map")
        ax.imshow(gridmap,cmap=plt.cm.gray_r)
        ax.axis("off")
        

        ax = fig.add_subplot(grid[-1, :])
        ax.set_title("Landmark")
        cat_name = landmark_config['name']
        cmap = landmark_config['color']
        img = [i for i in range(len(cat_name))]
        img = np.asarray(img).reshape(1,-1)
        annot = np.asarray(cat_name).reshape(1,-1)
        sns.heatmap(img, cmap=cmap,annot=annot, fmt = '', cbar=False,annot_kws={"size": 5})
        ax.axis('off')

        # plt.tight_layout()
        plt.show()



def draw_path(rrt,traj_image,path,NUM_WAYPOINT = 0,file_path=None,scene_name=None,query_name=None):
    traj_image = np.ascontiguousarray(traj_image, dtype=np.uint8)
    for idx in range(len(path)-1):
        prev_xz = rrt.rstate[path[idx]]
        next_xz = rrt.rstate[path[idx+1]]
        prev_w,prev_h = rrt.play_area.xz2coor(prev_xz[0],prev_xz[1])
        next_w,next_h = rrt.play_area.xz2coor(next_xz[0],next_xz[1])
        cv2.line(traj_image, (int(prev_w),int(prev_h)), (int(next_w),int(next_h)), (0,255,255), 2)
    # plt.imshow(traj_image)
    # plt.plot(path_wh[:,0], path_wh[:,1],'-c')

    # Cast start&end point coordinate in image
    # startx, startz = (rrt.start.x), (rrt.start.z)
    endx,   endz   = (rrt.end.x), (rrt.end.z)
    
    # startx, startz = rrt.play_area.xz2coor(startx, startz )
    endx,   endz   = rrt.play_area.xz2coor(endx,   endz   )
    
    # plt.plot(startx, startz,"xr")
    cv2.circle(traj_image, (int(endx),int(endz)), 5, (255,0,255), thickness=5)
    cv2.putText(traj_image, str(NUM_WAYPOINT), (int(endx),int(endz)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    # plt.plot(endx, endz, "xr")
    plt.figure()
    plt.imshow(traj_image)
    plt.savefig('{}/{}_{}_traj.png'.format(file_path,scene_name,query_name))
    # plt.show()
    return traj_image

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

