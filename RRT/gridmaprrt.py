"""

Path planning Sample Code with Randomized Rapidly-Exploring Random Trees (RRT)

author: AtsushiSakai(@Atsushi_twi)

"""

from difflib import get_close_matches
import math
from os import stat
import random

import matplotlib.pyplot as plt
import numpy as np
from ithor_tools.transform import cornerpoint_projection,ndarray

random.seed(42)
np.random.seed(42)

class RRT:
    """
    Class for RRT planning
    """

    class Node:
        """
        RRT Node
        """

        def __init__(self, x, y):
            self.x = x
            self.z = y
            self.path_x = []
            self.path_z = []
            self.parent = None

    class AreaBounds:

        def __init__(self, area):
            self.xmin = float(area[0])
            self.xmax = float(area[1])
            self.zmin = float(area[2])
            self.zmax = float(area[3])
    
        def xz2coor(self,x,z):
            w = (x-self.xmin)/(self.xmax-self.xmin) * (self.xmax_coor - self.xmin_coor) + self.xmin_coor
            h = (z-self.zmin)/(self.zmax-self.zmin) * (self.zmax_coor - self.zmin_coor) + self.zmin_coor
            
            return (w,h)

    def __init__(self,
                 controller,
                 expand_dis=0.1,
                 goal_sample_rate=5,
                 max_iter=500,
                 path_resolution = None,

                 ):
        """
        Setting Parameter

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        path_resolution: for latter path smoothing
        play_area:stay inside this area [xmin,xmax,ymin,ymax]

        """        
        # Set controller
        self.controller = controller
        self.set_rstate(controller)
        self.set_play_area()
        self.set_xz2coor_param()
        self.set_delta()
        
        if path_resolution is None:
            self.path_resolution = expand_dis/5
        else:
            self.path_resolution = path_resolution
        
        
        self.expand_dis = expand_dis
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.node_list = []
        
    
    def set_play_area(self):
        controller = self.controller
        scene_bounds = controller.last_event.metadata['sceneBounds']['cornerPoints']
        scene_bounds = cornerpoint_projection(scene_bounds)
        scene_bounds = ndarray(scene_bounds)
        
        x_max, z_max = np.max(scene_bounds,axis=0)
        x_min, z_min = np.min(scene_bounds,axis=0)
        
        play_area = [x_min,x_max,z_min,z_max]
        self.play_area = self.AreaBounds(play_area)
    
    def set_rstate(self,controller):
        controller.step(dict(action='GetReachablePositions'))
        rstate = controller.last_event.metadata['actionReturn']
        
        # import copy
        # rstate = copy.deepcopy(rstate)
        array_rstate = np.zeros((len(rstate),2))
        
        for idx in range(len(rstate)):
            state = rstate[idx]
            temp =  ndarray([state['x'],state['z']])
            array_rstate[idx,:] = temp
        
        self.rstate = array_rstate

    def set_goal(self,goal):
        goal = ndarray([goal['x'],goal['z']])
        idx,goal = self.get_closest_rstate(goal)
        
        self.end = self.Node(goal[0],goal[1])
        self.end.idx = idx

    def set_start(self,start):
        self.y_default = start['y']
        
        start = ndarray([start['x'],start['z']])
        idx,start = self.get_closest_rstate(start)
        
        self.start = self.Node(start[0],start[1])
        self.start.idx = idx
        
    def planning(self, animation=False, verbose = False):
        """
        rrt path planning

        animation: flag for animation on or off
        """

        self.node_list = [self.start]
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            if animation:
                self.draw_graph(rnd_node)
                
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            # Check Colision
            new_node,safety_flag = self.steer_collision(nearest_node, rnd_node, self.expand_dis,verbose=verbose)

            if safety_flag:
                self.node_list.append(new_node)

                if self.calc_dist_to_goal(self.node_list[-1].x,
                                        self.node_list[-1].z) <= self.expand_dis:
                    final_node,safety_flag = self.steer_collision(self.node_list[-1], self.end,
                                            self.expand_dis)
                    self.final_node = final_node
                    if safety_flag:
                        print("path found!")
                        final_path = self.generate_final_course()
                        self.final_path = final_path
                        return final_path

            # if animation and i % 5:
            #     self.draw_graph(rnd_node)

        print('Can not find path')
        return None  # cannot find path

    def steer_collision(self, from_node, to_node, extend_length=float("inf"),resolution=5, verbose=False):
        safty_flag = True
        
        new_node = self.Node(from_node.x, from_node.z)
        d, theta = self.calc_distance_and_angle(new_node, to_node)
        
        # saving path to animate & latter smoothing
        new_node.path_x = [new_node.x]
        new_node.path_z = [new_node.z]

        if extend_length > d:
            extend_length = d

        dep = ndarray([from_node.x, from_node.z])
        des = dep + ndarray([extend_length * math.cos(theta), extend_length * math.sin(theta)])
        
        path = np.linspace(dep, des, resolution)
        path = path[1:]
        for des_inter in path:
            idx, closest = self.get_closest_rstate(des_inter)
            
            ## If distance between des and closest is bigger than threshold. Treat as Collision
            if np.linalg.norm(des_inter-closest) > self.delta * 0.7:
                safty_flag = False
                return None, safty_flag
        
        new_node.x, new_node.z, new_node.idx = closest[0], closest[1], idx
        
        new_node.path_x, new_node.path_z = path[:,0], path[:,1]
        
        new_node.parent = from_node
        
        return new_node, safty_flag
    
    def get_Navigation_success_flag(self,path,verbose=False):
        
        # Go to initial point
        from_xz = self.rstate[path[0]]
        from_pos = dict(x=from_xz[0], y=self.y_default, z=from_xz[1])
        
        event = self.controller.step(
            action="Teleport",
            position=from_pos,
        )
        if not event.metadata['lastActionSuccess']:
            if verbose: print("Can not Teleport to inital point")
            return  None,False
        
        path = path[1:]
        for i,idx in enumerate(path):
            to_xz = self.rstate[idx]
            to_pos = dict(x=to_xz[0], y=self.y_default, z=to_xz[1])
            
            dx,dz = to_pos['x']-from_pos['x'], to_pos['z']-from_pos['z']
            trot = math.atan2(dx,dz)*180/math.pi
            crot = self.controller.last_event.metadata['agent']['rotation']['y'] 
            rot = trot - crot
            
            # RotateRight
            event = self.controller.step(
                action="RotateRight",
                degrees=rot
            )
            if not event.metadata['lastActionSuccess']:
                if verbose: print("Can not Rotate. Problem in {}th element".format(i))
                return  None,False
            
            # MoveAhead
            event = self.controller.step(
            action="MoveAhead",
            moveMagnitude=math.sqrt(dx**2+dz**2)
            )
            if not event.metadata['lastActionSuccess']:
                if verbose: print("Can not MoveAhead. Problem in {}th element".format(i))
                return None,False
            
            # Teleport to current position
            final_pos = self.controller.last_event.metadata['agent']['position']
            event = self.controller.step(
                action="Teleport",
                position=final_pos,
            )
            if not event.metadata['lastActionSuccess']:
                if verbose: print("Can not Teleport2. Problem in {}th element".format(i))
                return None,False

            from_pos = to_pos
        
        if verbose: print("Navigation Successful!")
        return final_pos,True
        
    def generate_final_course(self):
        goal_ind = len(self.node_list) - 1
        path = [self.end.idx]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append(node.idx)
            node = node.parent
        path.append(node.idx)

        path.reverse()
        return path

    def calc_dist_to_goal(self, x, z):
        dx = x - self.end.x
        dz = z - self.end.z
        return math.hypot(dx, dz)

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            if (self.rstate == None).any():
                # Get random sample from total space
                rnd = self.get_random_node_from_playarea()
            else:
                # Get random sample from reachable space
                rnd = self.get_random_node_from_rstate(noise=0)
                
        else:  # goal point sampling
            rnd = self.Node(self.end.x, self.end.z)
        return rnd
    
    def get_random_node_from_playarea(self):
        play_area = self.play_area
        node = self.Node(
            random.uniform(play_area.xmin, play_area.xmax),
            random.uniform(play_area.zmin, play_area.zmax))
        
        return node
    
    def get_random_node_from_rstate(self,noise=0):
        point = random.choice(self.rstate)
        
        x,z = point[0], point[1]
        
        # get noise to x,z
        noise1,noise2 = -1+2*np.random.rand(),-1+2*np.random.rand()
        noise1,noise2 = noise*noise1, noise*noise2
        
        x,z = x+noise1,z+noise2
        node = self.Node(x,z)

        return node

    def draw_graph(self, rnd=None):
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])

        if rnd is not None:
            w,h = self.play_area.xz2coor(rnd.x, rnd.z)
            plt.plot(w,h, "^k")
        
        # Cast rrt path to coordinate in image
        topview = self.controller.last_event.third_party_camera_frames[0]
            
        for node in self.node_list:
            if node.parent:
                node_path_x,node_path_z = ndarray(node.path_x), ndarray(node.path_z)
                node_path_w,node_path_h = self.play_area.xz2coor(node_path_x,node_path_z)
                plt.plot(node_path_w,node_path_h, "-g")
                # plt.show()
                
        if self.play_area is not None:
            wmin,hmin = self.play_area.xz2coor(self.play_area.xmin,self.play_area.zmin)
            wmax,hmax = self.play_area.xz2coor(self.play_area.xmax,self.play_area.zmax)
            plt.plot([wmin, wmax,wmax, wmin,wmin],
                     [hmin, hmin,hmax, hmax,hmin],
                     "-k")

        # Cast start&end point coordinate in image
        startx, startz = (self.start.x), (self.start.z)
        endx,   endz   = (self.end.x), (self.end.z)
        
        starth, startw = self.play_area.xz2coor(startx, startz )
        endh,   endw   = self.play_area.xz2coor(endx,   endz   )
        
        plt.plot(starth, startw,"xr")
        plt.plot(endh, endw, "xr")
        # plt.axis("equal")
        # plt.grid(True)
        
        plt.imshow(topview)
        plt.axis("off")
        plt.show()
        plt.pause(0.001)

    def plot_path(self, path):
        plt.clf()
        
        from_pos = dict(x=self.start.x, y=self.y_default, z=self.start.z)
        event = self.controller.step(
            action="Teleport",
            position=from_pos,
        )
        if not event.metadata['lastActionSuccess']:
            print("Can't go to start position")
            return None
        topview = self.controller.last_event.third_party_camera_frames[0]
        
        path_wh = []
        for idx in path:
            xz = self.rstate[idx]
            w,h = self.play_area.xz2coor(xz[0],xz[1])
            path_wh.append([w,h])
        path_wh = ndarray(path_wh)

        
        plt.plot(path_wh[:,0], path_wh[:,1],'-c')

        if self.play_area is not None:
            xmin,zmin = self.play_area.xz2coor(self.play_area.xmin,self.play_area.zmin)
            xmax,zmax = self.play_area.xz2coor(self.play_area.xmax,self.play_area.zmax)
            plt.plot([xmin, xmax,xmax, xmin,xmin],
                     [zmin, zmin,zmax, zmax,zmin],
                     "-k")


        # Cast start&end point coordinate in image
        startx, startz = (self.start.x), (self.start.z)
        endx,   endz   = (self.end.x), (self.end.z)
        
        startx, startz = self.play_area.xz2coor(startx, startz )
        endx,   endz   = self.play_area.xz2coor(endx,   endz   )
        
        plt.plot(startx, startz,"xr")
        plt.plot(endx, endz, "xr")
        
        plt.imshow(topview)
        plt.axis("off")
        plt.show()
        plt.pause(0.001)


    def set_xz2coor_param(self):
        topview = self.controller.last_event.third_party_camera_frames[0]
        
        
        for zmax_coor in range(topview.shape[0]):
            
            temp = topview[zmax_coor,:] == 255*np.ones_like(topview[zmax_coor,:])
            if not temp.all():
                break
        self.play_area.zmax_coor = zmax_coor
        
        for zmin_coor in range(topview.shape[0]).__reversed__():
            
            temp = topview[zmin_coor,:] == 255*np.ones_like(topview[zmin_coor,:])
            if not temp.all():
                break
        self.play_area.zmin_coor = zmin_coor
        
        for xmin_coor in range(topview.shape[1]):
            
            temp = topview[:,xmin_coor] == 255*np.ones_like(topview[:,xmin_coor])
            if not temp.all():
                break
        self.play_area.xmin_coor = xmin_coor

        for xmax_coor in range(topview.shape[1]).__reversed__():
            
            temp = topview[:,xmax_coor] == 255*np.ones_like(topview[:,xmax_coor])
            if not temp.all():
                break
        self.play_area.xmax_coor = xmax_coor
        
    @staticmethod
    def plot_circle(x, z, size, color="-b"):  # pragma: no cover
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
        zl = [z + size * math.sin(np.deg2rad(d)) for d in deg]
        plt.plot(xl, zl, color)

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        dlist = [(node.x - rnd_node.x)**2 + (node.z - rnd_node.z)**2
                 for node in node_list]
        minind = dlist.index(min(dlist))

        return minind

    @staticmethod
    def check_if_outside_play_area(node, play_area):

        if play_area is None:
            return True  # no play_area was defined, every pos should be ok

        if node.x < play_area.xmin or node.x > play_area.xmax or \
           node.z < play_area.zmin or node.z > play_area.zmax:
            return False  # outside - bad
        else:
            return True  # inside - ok

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dz = to_node.z - from_node.z
        d = math.hypot(dx, dz)
        theta = math.atan2(dz, dx)
        return d, theta
    
    def get_closest_rstate(self,state):
        assert len(state) == 2
        
        rstate = self.rstate
        
        state = ndarray(state)
        final = np.linalg.norm(rstate - state,axis=1)

        idx= np.argmin(final)
        
        state = rstate[idx]
        return idx, state
    
    def set_delta(self):
        rstate = self.rstate
        baseline = rstate[0]
        rstate = rstate[1:]


        final = np.linalg.norm(rstate - baseline,axis=1)
        idx= np.argmin(final)

        state = rstate[idx]

        delta = np.linalg.norm(state-baseline)
        
        self.delta = delta