"""

Path planning Sample Code with Randomized Rapidly-Exploring Random Trees (RRT)

author: AtsushiSakai(@Atsushi_twi)

"""

import math
import random

import matplotlib.pyplot as plt
import numpy as np

show_animation = True

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
            self.y = y
            self.path_x = []
            self.path_y = []
            self.parent = None

    class AreaBounds:

        def __init__(self, area):
            self.xmin = float(area[0])
            self.xmax = float(area[1])
            self.ymin = float(area[2])
            self.ymax = float(area[3])


    def __init__(self,
                 start,
                 goal,
                 gridmap,
                 w_quan,
                 h_quan,
                 rstate = None,
                 expand_dis=1.0,
                 stepsize=0.5,
                 goal_sample_rate=5,
                 max_iter=500,
                 play_area=None,
                 robot_radius=0,
                 path_resolution = None
                 ):
        """
        Setting Parameter

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        gridmap: For [x,y] Occupancy gridmap [[1,][1,],...]
        randArea:Random Sampling Area [min,max]
        play_area:stay inside this area [xmin,xmax,ymin,ymax]
        robot_radius: robot body modeled as circle with given radius

        """
        
        self.start = self.Node(start[0], start[1])
        self.end = self.Node(goal[0], goal[1])
        self.w_quan, self.h_quan = w_quan, h_quan
        
        if play_area is not None:
            self.play_area = self.AreaBounds(play_area)
        else:
            self.play_area = None
        
        if path_resolution is None:
            path_resolution = expand_dis/5
        
        self.stepsize = stepsize
        self.robot_radius = int(robot_radius//stepsize)
        self.expand_dis = expand_dis/stepsize
        self.path_resolution = path_resolution/stepsize
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.gridmap = gridmap
        self.node_list = []
        self.rstate = rstate
    def planning(self, animation=False):
        """
        rrt path planning

        animation: flag for animation on or off
        """

        self.node_list = [self.start]
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if self.check_if_outside_play_area(new_node, self.play_area) and \
               self.check_collision(
                   new_node, self.gridmap, self.robot_radius):
                self.node_list.append(new_node)

            if animation and i % 5 == 0:
                self.draw_graph(rnd_node)

            if self.calc_dist_to_goal(self.node_list[-1].x,
                                      self.node_list[-1].y) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.end,
                                        self.expand_dis)
                if self.check_collision(
                        final_node, self.gridmap, self.robot_radius):
                    print("path found!")
                    return self.generate_final_course(len(self.node_list) - 1)

            if animation and i % 5:
                self.draw_graph(rnd_node)

        print('Can not find path')
        return None  # cannot find path

    def steer(self, from_node, to_node, extend_length=float("inf")):

        new_node = self.Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)

        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]

        if extend_length > d:
            extend_length = d

        n_expand = math.floor(extend_length / self.path_resolution)

        for _ in range(n_expand):
            new_node.x += self.path_resolution * math.cos(theta)
            new_node.y += self.path_resolution * math.sin(theta)
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)

        d, _ = self.calc_distance_and_angle(new_node, to_node)
        if d <= self.path_resolution:
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)
            new_node.x = to_node.x
            new_node.y = to_node.y

        new_node.parent = from_node

        return new_node

    def generate_final_course(self, goal_ind):
        path = [[self.end.x, self.end.y]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])

        return path

    def calc_dist_to_goal(self, x, y):
        dx = x - self.end.x
        dy = y - self.end.y
        return math.hypot(dx, dy)

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            if self.rstate == None:
                # Get random sample from total space
                rnd = self.Node(
                    random.uniform(0, self.w_quan),
                    random.uniform(0, self.h_quan))
            else:
                # Get random sample from reachable space
                point = random.choice(self.rstate)
                # point = np.array(point)
                point = np.array(point) + (-1+2*np.random.rand())
                
                w,h = point
                rnd = self.Node(w,h)
        else:  # goal point sampling
            rnd = self.Node(self.end.x, self.end.y)
        return rnd

    def draw_graph(self, rnd=None):
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        if rnd is not None:
            plt.plot(rnd.y, rnd.x, "^k")
            if self.robot_radius > 0.0:
                self.plot_circle(rnd.y, rnd.x,self.robot_radius, '-r')
        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_y,node.path_x, "-g")

        # for (ox, oy, size) in self.gridmap:
        #     self.plot_circle(ox, oy, size)

        if self.play_area is not None:
            plt.plot([self.play_area.xmin, self.play_area.xmax,
                      self.play_area.xmax, self.play_area.xmin,
                      self.play_area.xmin],
                     [self.play_area.ymin, self.play_area.ymin,
                      self.play_area.ymax, self.play_area.ymax,
                      self.play_area.ymin],
                     "-k")

        plt.plot(self.start.y, self.start.x,"xr")
        plt.plot(self.end.y, self.end.x, "xr")
        plt.axis("equal")
        plt.grid(True)
        
        plt.imshow(self.gridmap, cmap='gray')
        plt.show()
        plt.pause(0.001)

    @staticmethod
    def plot_circle(x, y, size, color="-b"):  # pragma: no cover
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
        yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
        plt.plot(xl, yl, color)

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        dlist = [(node.x - rnd_node.x)**2 + (node.y - rnd_node.y)**2
                 for node in node_list]
        minind = dlist.index(min(dlist))

        return minind

    @staticmethod
    def check_if_outside_play_area(node, play_area):

        if play_area is None:
            return True  # no play_area was defined, every pos should be ok

        if node.x < play_area.xmin or node.x > play_area.xmax or \
           node.y < play_area.ymin or node.y > play_area.ymax:
            return False  # outside - bad
        else:
            return True  # inside - ok

    @staticmethod
    def check_collision(node, gridmap, robot_radius):

        if node is None:
            return False
        
        x,y = node.x,node.y
        x,y = int(x),int(y)
        
        for x_neighbor in range(x-robot_radius,x+robot_radius+1):
            for y_neighbor in range(y-robot_radius,y+robot_radius+1):        
                if gridmap[x_neighbor,y_neighbor] == 1:
                    return False
        
        # for (ox, oy, size) in obstacleList:
        #     dx_list = [ox - x for x in node.path_x]
        #     dy_list = [oy - y for y in node.path_y]
        #     d_list = [dx * dx + dy * dy for (dx, dy) in zip(dx_list, dy_list)]

        #     if min(d_list) <= (size+robot_radius)**2:
        #         return False  # collision

        return True  # safe

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta

#     ## Code for smoothing
#     def path_smoothing(self,path, max_iter):
#         gridmap = self.gridmap
#         le = get_path_length(path)
#         for i in range(max_iter):
#             # Sample two points
#             pickPoints = [random.uniform(0, le), random.uniform(0, le)]
#             pickPoints.sort()
#             first = get_target_point(path, pickPoints[0])
#             second = get_target_point(path, pickPoints[1])

#             if first[2] <= 0 or second[2] <= 0:
#                 continue

#             if (second[2] + 1) > len(path):
#                 continue

#             if second[2] == first[2]:
#                 continue

#             # collision check
#             if not self.line_collision_check(first, second, gridmap, self.robot_radius):
#                 continue

#             # Create New path
#             print("new point!!")
#             newPath = []
#             newPath.extend(path[:first[2] + 1])
#             newPath.append([first[0], first[1]])
#             newPath.append([second[0], second[1]])
#             newPath.extend(path[second[2] + 1:])
#             path = newPath
#             le = get_path_length(path)

#         return path
    
#     def line_collision_check(self,first, second, gridmap,robot_radius):
#         # Line Equation

#         resolution = 10
#         x1 = first[0]
#         y1 = first[1]
#         x2 = second[0]
#         y2 = second[1]

#         p1 = np.array([x1,y1])
#         p2 = np.array([x2,y2])
#         print(p1,p2)
#         points = np.linspace(p1,p2,resolution)
#         for point in points:
#             x,y = point
#             node = self.Node(x,y)
#             if self.check_collision(node,gridmap,robot_radius):
#                 return False

#         return True
# def get_path_length(path):
#     le = 0
#     for i in range(len(path) - 1):
#         dx = path[i + 1][0] - path[i][0]
#         dy = path[i + 1][1] - path[i][1]
#         d = math.sqrt(dx * dx + dy * dy)
#         le += d

#     return le


# def get_target_point(path, targetL):
#     le = 0
#     ti = 0
#     lastPairLen = 0
#     for i in range(len(path) - 1):
#         dx = path[i + 1][0] - path[i][0]
#         dy = path[i + 1][1] - path[i][1]
#         d = math.sqrt(dx * dx + dy * dy)
#         le += d
#         if le >= targetL:
#             ti = i - 1
#             lastPairLen = d
#             break

#     partRatio = (le - targetL) / lastPairLen

#     x = path[ti][0] + (path[ti + 1][0] - path[ti][0]) * partRatio
#     y = path[ti][1] + (path[ti + 1][1] - path[ti][1]) * partRatio

#     return [x, y, ti]
    
    
### Example Run code ###
'''
def main(gx=6.0, gy=10.0):
    print("start " + __file__)

    # ====Search Path with RRT====
    obstacleList = [(5, 5, 1), (3, 6, 2), (3, 8, 2), (3, 10, 2), (7, 5, 2),
                    (9, 5, 2), (8, 10, 1)]  # [x, y, radius]
    # Set Initial parameters
    rrt = RRT(
        start=[0, 0],
        goal=[gx, gy],
        rand_area=[-2, 15],
        obstacle_list=obstacleList,
        # play_area=[0, 10, 0, 14]
        robot_radius=0.8
        )
    path = rrt.planning(animation=show_animation)

    if path is None:
        print("Cannot find path")
    else:
        print("found path!!")

        # Draw final path
        if show_animation:
            rrt.draw_graph()
            plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
            plt.grid(True)
            plt.pause(0.01)  # Need for Mac
            plt.show()


if __name__ == '__main__':
    main()
    
'''