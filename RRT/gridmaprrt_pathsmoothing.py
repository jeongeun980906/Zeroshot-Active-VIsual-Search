"""

Path planning Sample Code with RRT with path smoothing

@author: AtsushiSakai(@Atsushi_twi)

"""

from csv import excel_tab
import math
import os
import random
import sys
import numpy as np

import matplotlib.pyplot as plt

from RRT import gridmaprrt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from gridmaprrt import RRT
except ImportError:
    raise

random.seed(42)
np.random.seed(42)

def get_path_length(rstate,path):
    
    le = 0
    
    for i in range(len(path) - 1):
        from_xz,to_xz = rstate[path[i]],rstate[path[i+1]]
        
        dx = to_xz[0] - from_xz[0]
        dz = to_xz[1] - from_xz[1]
        d = math.sqrt(dx * dx + dz * dz)
        le += d

    return le



def get_target_point(path, targetL,rrtplanner,verbose,resolution=100):
    rstate = rrtplanner.rstate
    
    le = 0
    ti = 0
    lastPairLen = 0
    for i in range(len(path) - 1):
        from_xz,to_xz = rstate[path[i]],rstate[path[i+1]]
        
        dx = to_xz[0] - from_xz[0]
        dz = to_xz[1] - from_xz[1]
        d = math.sqrt(dx * dx + dz * dz)
        le += d
        if le >= targetL:
            ti = i - 1
            lastPairLen = d
            break
    
    ## Get Interpolated new point

    partRatio = (le - targetL) / lastPairLen

    from_xz,to_xz = rstate[path[ti]],rstate[path[ti+1]]

    inter_xz = np.array([to_xz[0] * partRatio + from_xz[0] * (1-partRatio),
                     to_xz[1] * partRatio + from_xz[1] * (1-partRatio)])
    
    
    inter_idx, inter_xz   = rrtplanner.get_closest_rstate(inter_xz)
    
    
    from_node  = rrtplanner.Node(from_xz[0],from_xz[1])
    inter_node = rrtplanner.Node(inter_xz[0],inter_xz[1])
    to_node    = rrtplanner.Node(to_xz[0],  to_xz[1]  )
    
    _, safty_flag1 = rrtplanner.steer_collision(from_node, inter_node, extend_length=float("inf"),verbose=False, resolution= resolution)
    _, safty_flag2 = rrtplanner.steer_collision(inter_node, to_node, extend_length=float("inf"),verbose=False, resolution= resolution)
    safty_flag = safty_flag1 and safty_flag2
    if not safty_flag:
        return None, safty_flag
    
    to_xz = np.array([to_node.x,to_node.z])
    
    ## Get closest index
    to_idx,to_xz   = rrtplanner.get_closest_rstate(to_xz)
    fr_idx,from_xz = rrtplanner.get_closest_rstate(from_xz)

    return [fr_idx,to_idx, ti], safty_flag

def line_collision_check(first, second,rrtplanner,verbose,resolution=100):

    rstate = rrtplanner.rstate
    from_xz,to_xz = rstate[first[1]],rstate[second[1]]
    
    from_node = rrtplanner.Node(from_xz[0],from_xz[1])
    to_node   = rrtplanner.Node(to_xz[0],  to_xz[1]  )
    
    _, safty_flag = rrtplanner.steer_collision(from_node, to_node, extend_length=float("inf"),verbose=verbose, resolution=resolution)
    
    
    return safty_flag

def path_smoothing(rrtplanner, max_iter,verbose=False):
    path = rrtplanner.final_path
    rstate = rrtplanner.rstate
    random.seed(42)
    np.random.seed(42)
    for i in range(max_iter):
        # Sample two points
        le = get_path_length(rstate,path)

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
        if verbose: print("new point!")
        newPath = []
        newPath.extend(path[:first[2] + 1])
        newPath.append(first[1])
        newPath.append(second[1])
        newPath.extend(path[second[2] + 2:])
        path = newPath
      
    newPath = []
    newPath.append(path[0])
    path = path[1:]
    for ele in path:
        if not(newPath[-1] == ele):
            newPath.append(ele)
    path = newPath
    return path



