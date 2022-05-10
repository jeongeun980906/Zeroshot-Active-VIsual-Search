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


def get_target_point(path, targetL):
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

    partRatio = (le - targetL) / lastPairLen

    x = path[ti][0] + (path[ti + 1][0] - path[ti][0]) * partRatio
    y = path[ti][1] + (path[ti + 1][1] - path[ti][1]) * partRatio

    return [x, y, ti]


def line_collision_check(first, second, gridmap,margin):
    # Line Equation
    margin = margin
    
    resolution = 10
    x1 = first[0]
    y1 = first[1]
    x2 = second[0]
    y2 = second[1]

    p1 = np.array([x1,y1])
    p2 = np.array([x2,y2])
    
    points = np.linspace(p1,p2,resolution)
    for point in points:
        x,y = point
        x,y = int(x), int(y)
        
        for x_neighbor in range(x-margin,x+margin+1):
            for y_neighbor in range(y-margin, y+margin+1):
                if gridmap[x_neighbor,y_neighbor]==1:
                    return False

    return True  # OK


def path_smoothing(path, max_iter, gridmap,margin=4):
    le = get_path_length(path)

    for i in range(max_iter):
        # Sample two points
        pickPoints = [random.uniform(0, le), random.uniform(0, le)]
        pickPoints.sort()
        first = get_target_point(path, pickPoints[0])
        second = get_target_point(path, pickPoints[1])

        if first[2] <= 0 or second[2] <= 0:
            continue

        if (second[2] + 1) > len(path):
            continue

        if second[2] == first[2]:
            continue

        # collision check
        if not line_collision_check(first, second, gridmap,margin):
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

    return path


### Example Run code ###
'''
def main():
    # ====Search Path with RRT====
    # Parameter
    obstacleList = [
        (5, 5, 1),
        (3, 6, 2),
        (3, 8, 2),
        (3, 10, 2),
        (7, 5, 2),
        (9, 5, 2)
    ]  # [x,y,size]
    rrt = RRT(start=[0, 0], goal=[6, 10],
              rand_area=[-2, 15], obstacle_list=obstacleList)
    path = rrt.planning(animation=show_animation)

    # Path smoothing
    maxIter = 1000
    smoothedPath = path_smoothing(path, maxIter, obstacleList)

    # Draw final path
    if show_animation:
        rrt.draw_graph()
        plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')

        plt.plot([x for (x, y) in smoothedPath], [
            y for (x, y) in smoothedPath], '-c')

        plt.grid(True)
        plt.pause(0.01)  # Need for Mac
        plt.show()

if __name__ == '__main__':
    main()
    
'''
