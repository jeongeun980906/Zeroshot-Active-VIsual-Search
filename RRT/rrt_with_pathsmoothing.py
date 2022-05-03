"""

Path planning Sample Code with RRT with path smoothing

@author: AtsushiSakai(@Atsushi_twi)

"""

import math
import os
import random
import sys
import numpy as np

import matplotlib.pyplot as plt

from RRT import rrt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from rrt import RRT
except ImportError:
    raise

show_animation = True

random.seed(42)
def get_path_length(path):
    le = 0
    for i in range(len(path) - 1):
        dx = path[i + 1][0] - path[i][0]
        dy = path[i + 1][1] - path[i][1]
        d = math.sqrt(dx * dx + dy * dy)
        le += d

    return le


def get_target_point(path, targetL,rrtplanner,verbose):
    le = 0
    ti = 0
    lastPairLen = 0
    for i in range(len(path) - 1):
        dx = path[i + 1][0] - path[i][0]
        dy = path[i + 1][1] - path[i][1]
        d = math.sqrt(dx * dx + dy * dy)
        le += d
        if le >= targetL:
            ti = i - 1
            lastPairLen = d
            break
    
    y_default = rrtplanner.y_default
    
    partRatio = (le - targetL) / lastPairLen
    # partRatio = 1

    to_x = path[ti][0] + (path[ti + 1][0] - path[ti][0]) * partRatio
    to_z = path[ti][1] + (path[ti + 1][1] - path[ti][1]) * partRatio

    from_pos = dict(x=path[ti][0], y=y_default, z=path[ti][1])
    to_pos = dict(x=to_x, y=y_default, z=to_z)
    
    final_pos, safty_flag = rrtplanner.get_Navigation_success_flag(from_pos,to_pos,verbose)

    if not safty_flag:
        return None, safty_flag
    
    # event = rrtplanner.controller.step(
    #     action = "Teleport",
    #     position = final_pos
    # )
    
    # if not event.metadata['lastActionSuccess']:
    #     print("Somthing Wrong 2")
    #     return  False
    
    x,z = final_pos['x'],final_pos['z']

    return [x, z, ti], safty_flag
    


def line_collision_check(first, second,rrtplanner,verbose):
    # Line Equation
    y_default = rrtplanner.y_default
    from_pos = dict(x=first[0] , y=y_default, z=first[1] )
    to_pos   = dict(x=second[0], y=y_default, z=second[1])
    
    final_pos, safty_flag = rrtplanner.get_Navigation_success_flag(from_pos,to_pos,verbose=verbose)
    
    # event = rrtplanner.controller.step(
    #     action = "Teleport",
    #     position = final_pos
    # )
    
    # if not event.metadata['lastActionSuccess']:
    #     print("Somthing Wrong 1")
    #     return  False
    
    return safty_flag


def path_smoothing(rrtplanner, max_iter,verbose=False):
    path = rrtplanner.final_path
    le = get_path_length(path)

    for i in range(max_iter):
        # Sample two points
        pickPoints = [random.uniform(0, le), random.uniform(0, le)]
        pickPoints.sort()
        first,flag1 = get_target_point(path, pickPoints[0],rrtplanner,verbose=verbose)
        second,flag2 = get_target_point(path, pickPoints[1],rrtplanner,verbose=verbose)
        
        if (not flag1) or (not flag2):
            if verbose: print("Collision During Interpolation")
            continue
        
        if first[2] <= 0 or second[2] <= 0:
            continue

        if (second[2] + 1) > len(path):
            continue

        if second[2] == first[2]:
            continue

        # collision check
        if not line_collision_check(first, second, rrtplanner,verbose=verbose):
            if verbose: print("Collision in two connecting")
            continue

        # Create New path
        # print("new point!")
        newPath = []
        newPath.extend(path[:first[2] + 1])
        newPath.append([first[0], first[1]])
        newPath.append([second[0], second[1]])
        newPath.extend(path[second[2] + 1:])
        path = newPath
        le = get_path_length(path)

    path = np.array(path)
    path = np.flip(path,0)
    return path

