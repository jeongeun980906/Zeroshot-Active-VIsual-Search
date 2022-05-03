"""

Path planning Sample Code with Randomized Rapidly-Exploring Random Trees (RRT)

author: AtsushiSakai(@Atsushi_twi)

"""

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
        self.rstate = rstate

    def set_goal(self,goal):
        x,z = goal['x'],goal['z']
        self.end = self.Node(x,z)
        
    def set_start(self,start):
        x,z = start['x'],start['z']
        self.y_default = start['y']
        self.start = self.Node(x,z)
        
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
                        final_path = self.generate_final_course(len(self.node_list) - 1)
                        self.final_path = final_path
                        return final_path

            # if animation and i % 5:
            #     self.draw_graph(rnd_node)

        print('Can not find path')
        return None  # cannot find path

    def steer_collision(self, from_node, to_node, extend_length=float("inf"),verbose=False):    
        new_node = self.Node(from_node.x, from_node.z)
        d, theta = self.calc_distance_and_angle(new_node, to_node)
        
        # make from_pos
        from_pos = dict(x=from_node.x, y=self.y_default, z=from_node.z)
        
        # saving path to animate & latter smoothing
        new_node.path_x = [new_node.x]
        new_node.path_z = [new_node.z]

        if extend_length > d:
            extend_length = d

        n_expand = math.floor(extend_length / self.path_resolution)

        for _ in range(n_expand):
            new_node.x += self.path_resolution * math.cos(theta)
            new_node.z += self.path_resolution * math.sin(theta)
            new_node.path_x.append(new_node.x)
            new_node.path_z.append(new_node.z)

        d, _ = self.calc_distance_and_angle(new_node, to_node)
        if d <= self.path_resolution:
            new_node.path_x.append(to_node.x)
            new_node.path_z.append(to_node.z)
            new_node.x = to_node.x
            new_node.z = to_node.z
        
        new_node.parent = from_node
        
        
        
        to_pos = dict(x=new_node.x, y=self.y_default, z=new_node.z)
        
        final_pos, safty_flag = self.get_Navigation_success_flag(from_pos,to_pos,verbose)
        if not safty_flag:
            return _, False
        
        final_node = self.Node(final_pos['x'],final_pos['z'])
        
        new_node.path_x.append(final_node.x)
        new_node.path_z.append(final_node.z)
        new_node.x = final_node.x
        new_node.z = final_node.z

        
        return new_node, safty_flag
    def get_Navigation_success_flag(self,from_pos,to_pos,verbose=False):
        event = self.controller.step(
            action="Teleport",
            position=from_pos,
        )
        if not event.metadata['lastActionSuccess']:
            if verbose: print("Can not Teleport1")
            return None, False
        
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
            if verbose: print("Can not Rotate")
            return None, False
        
        # MoveAhead
        event = self.controller.step(
        action="MoveAhead",
        moveMagnitude=math.sqrt(dx**2+dz**2)
        )
        if not event.metadata['lastActionSuccess']:
            if verbose: print("Can not MoveAhead")
            return None, False
        
        # Teleport to current position
        final_pos = self.controller.last_event.metadata['agent']['position']
        event = self.controller.step(
            action="Teleport",
            position=final_pos,
        )
        if not event.metadata['lastActionSuccess']:
            if verbose: print("Can not Teleport2")
            return None, False
        
        return final_pos,True
        
    def generate_final_course(self, goal_ind):
        path = [[self.end.x, self.end.z]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.z])
            node = node.parent
        path.append([node.x, node.z])

        return path

    def calc_dist_to_goal(self, x, z):
        dx = x - self.end.x
        dz = z - self.end.z
        return math.hypot(dx, dz)

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            if self.rstate == None:
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
        
        x,z = point['x'], point['z']
        
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
            x,z = self.play_area.xz2coor(rnd.x, rnd.z)
            plt.plot(x,z, "^k")
        
        # Cast rrt path to coordinate in image
        topview = self.controller.last_event.third_party_camera_frames[0]
            
        for node in self.node_list:
            if node.parent:
                node_path_x,node_path_z = ndarray(node.path_x), ndarray(node.path_z)
                node_path_x,node_path_z = self.play_area.xz2coor(node_path_x,node_path_z)
                plt.plot(node_path_x,node_path_z, "-g")

        if self.play_area is not None:
            plt.plot([self.play_area.xmin, self.play_area.xmax,
                      self.play_area.xmax, self.play_area.xmin,
                      self.play_area.xmin],
                     [self.play_area.zmin, self.play_area.zmin,
                      self.play_area.zmax, self.play_area.zmax,
                      self.play_area.zmin],
                     "-k")

        # Cast start&end point coordinate in image
        startx, startz = (self.start.x), (self.start.z)
        endx,   endz   = (self.end.x), (self.end.z)
        
        startx, startz = self.play_area.xz2coor(startx, startz )
        endx,   endz   = self.play_area.xz2coor(endx,   endz   )
        
        plt.plot(startx, startz,"xr")
        plt.plot(endx, endz, "xr")
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
        
        node_path_x, node_path_z = np.array(path)[:,0],np.array(path)[:,1]
        node_path_x, node_path_z = self.play_area.xz2coor(node_path_x,node_path_z)
        plt.plot(node_path_x, node_path_z,'-c')

        if self.play_area is not None:
            plt.plot([self.play_area.xmin, self.play_area.xmax,
                      self.play_area.xmax, self.play_area.xmin,
                      self.play_area.xmin],
                     [self.play_area.zmin, self.play_area.zmin,
                      self.play_area.zmax, self.play_area.zmax,
                      self.play_area.zmin],
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
