from ai2thor.util.metrics import (
    get_shortest_path_to_point
)
import numpy as np
from eval_ithor.reset import move_init
import math

def move_waypoint(controller,waypoint):
    controller.step(action="Teleport",
                    position = waypoint[0], rotation = dict(x=0,y=waypoint[1],z=0))
    pos = controller.last_event.metadata['agent']['position']
    pos = [pos['x'],pos['z']]
    print(controller.last_event.metadata['lastActionSuccess'])
    return pos


class co_occurance_based_schedular():
    def __init__(self,landmarks,visible_landmark_name):
        self.landmarks = landmarks
        self.visible_landmark_name = visible_landmark_name

    def get_node(self,scenemap,controller,co_occurance_score,thres):
        cpos = controller.last_event.metadata['agent']['position']
        node_info = [(cpos,0)]
        self.node = [-1]
        for e,score in enumerate(co_occurance_score):
            landmark_name = self.visible_landmark_name[e]
            if score>thres:
                for e,l in enumerate(self.landmarks):
                    if landmark_name == l['name']:
                        lois = scenemap.landmark_loi[e]
                        for loi in lois:
                            goal_point = loi[0]
                            goal_rot = loi[1]
                            node_info.append([goal_point,goal_rot,score])
                            self.node.append(score)
        self.node_info = node_info
        

    def get_edge(self,controller):
        self.edge = np.zeros((len(self.node_info),len(self.node_info)))
        for i in range(len(self.node_info)):
            for j in range(len(self.node_info)):
                if j>=i: 
                    break
                pos1 = self.node_info[i][0]
                pos2 = self.node_info[j][0]
                dis = self.shortest_path_length(controller,pos1,pos2)
                self.edge[i,j] = dis
                self.edge[j,i] = dis
        

    def shortest_path_length(self,controller,goal,init):
        try:
            path = get_shortest_path_to_point(
                    controller=controller,
                        target_position= goal,
                        initial_position= init,
                        allowed_error=0.01
                    )
        except:
            return 100
        distance = 0
        for e in range(len(path)-1):
            dx = abs(path[e]['x'] -  path[e+1]['x'])
            dz = abs(path[e]['z'] - path[e+1]['z'])
            distance += math.sqrt(dx**2+dz**2)
        return distance

    def optimize(self):
        index = [0]
        distance = self.edge
        score = (1+1e-3-np.asarray(self.node))/2
        for i in range(1,len(self.node_info)):
            temp = np.arange(len(self.node_info))
            temp = np.delete(temp,index)
            dis = distance.copy()[index[-1]] # [N-i x 1]
            dis = np.delete(dis,index) # [N-i-1 x 1]
            scaled_score = np.delete(score,index)
            scaled_score = dis*scaled_score # [N -i -1 x 1]
            new = np.argmin(scaled_score)
            new = temp[new]
            index.append(new)
        print(index)
        path = []
        for idx in index:
            path.append(self.node_info[idx])
        return path
