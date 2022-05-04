import numpy as np
import matplotlib.pyplot as plt
import copy
# from RRT import rrt as rrts

def giveMargintoGridmap(grid_map,wh_quan,margin_quan):
    ToGiveMargin = []
    print(margin_quan)
    w_quan,h_quan = wh_quan
    for w in range(w_quan):
        for h in range(h_quan):
            if grid_map[w,h,0] == 0:
                # if (x,y) is occupied
                
                # make it's neighbor occuppied with with margin
                for w_margin in range(2*margin_quan+1):
                    for h_margin in range(2*margin_quan+1):
                        w_neighbor = w + w_margin - margin_quan
                        h_neighbor = h + h_margin - margin_quan
                        if 0<=w_neighbor and w_neighbor<w_quan and 0<=h_neighbor and h_neighbor<h_quan:
                            ToGiveMargin.append((w_neighbor,h_neighbor))
                        
    ToGiveMargin = set(ToGiveMargin)
    ToGiveMargin = [list(ele) for ele in ToGiveMargin]
    
    for ele in ToGiveMargin:
        w,h = ele
        grid_map[w,h,:] = [0,0,0]

    return grid_map

class single_scenemap():
    def __init__(self,scenebound, reachable_state, landmark_names,landmarks,
                                stepsize=0.25, margin=0):
        scenebound = np.asarray(scenebound)
        x_max, z_max = np.max(scenebound,axis=0)
        x_min, z_min  = np.min(scenebound,axis=0)
        print(x_min,x_max,z_min,z_max)
        self.stepsize = stepsize
        x_max = self.stepsize* (x_max//self.stepsize)
        z_max = self.stepsize* (z_max//self.stepsize)
        x_min = self.stepsize* (x_min//self.stepsize +1)
        z_min = self.stepsize* (z_min//self.stepsize +1)

        x_len =  x_max- x_min
        z_len =  z_max- z_min
        # print(x_min,x_max,z_min,z_max)
        self.x_min, self.x_max = x_min, x_max
        self.z_min, self.z_max = z_min, z_max
        self.y_default = reachable_state[0]['y']
        w_quan = int(x_len//self.stepsize)+1
        h_quan = int(z_len//self.stepsize)+1
        
        self.w_quan = w_quan
        self.h_quan = h_quan
        
        self.grid_map = np.zeros((w_quan,h_quan,3))

        self.landmark_names = landmark_names
        self.landmarks = landmarks

        self.landmark_colors = plt.cm.get_cmap('Set3', len(landmark_names))
        self.get_gridmap(reachable_state,margin)
        
        self.get_rstate(reachable_state)
        
    def get_gridmap(self,reachable_state,margin):
        rstate = [[r['x'],r['z']] for r in reachable_state]
        rstate = np.asarray(rstate) # [N x 2]
        rstate[:,0] -= self.x_min
        rstate[:,1] -= self.z_min
        rstate /= self.stepsize
        rstate = rstate.astype('int32')
        for r in rstate:
            self.grid_map[r[0],r[1],:] =[1,1,1]
        w_quan = self.w_quan
        h_quan = self.h_quan
        
        # Give margin to gridmap
        margin_quan = int(margin//self.stepsize)
        self.grid_map = giveMargintoGridmap(self.grid_map,(w_quan,h_quan), margin_quan)
        self.plot_landmarks()
        
    def plot_landmarks(self):
        for l in self.landmarks:
            pos = self.xyz2grid(l['cp'])
            color = self.landmark_names.index(l['name'])
            self.grid_map[pos[0],pos[1],:] = self.landmark_colors(color)[:3]

    def get_landmark_viewpoint(self,pos,landmark_name,controller):
        size = int(self.stepsize*50)
        i = 1
        [x,y] = self.xyz2grid(pos)
        while True:
            if self.grid_map[x+i,y,0]*self.grid_map[x-i,y,0]:
                if self.grid_map[x+i+size,y,0] or self.grid_map[x-i-size,y,0]:
                    cpos = controller.last_event.metadata['agent']['position']
                    crot = controller.last_event.metadata['agent']['rotation']
                    controller.step("Teleport", position = self.grid2xyz([x+i+size,y],0.91), rotation =  270
                                )
                    temp = controller.last_event.metadata['objects']
                    for t in temp:
                        if t == landmark_name:
                            print(t['Visible'])
                            if t['Visible']:
                                controller.step("Teleport", position = cpos, rotation =  crot
                                )
                                return self.grid2xyz([x+i+size,y],0.91), 270
                    controller.step("Teleport", position = cpos, rotation =  crot
                                )
                    return self.grid2xyz([x-i+size,y],0.91), 90
            if self.grid_map[x,y+i,0]*self.grid_map[x,y-i,0]:
                if self.grid_map[x,y+i+size,0] or self.grid_map[x,y-i-size,0]:
                    cpos = controller.last_event.metadata['agent']['position']
                    crot = controller.last_event.metadata['agent']['rotation']
                    controller.step("Teleport", position = self.grid2xyz([x,y+i+size],0.91), rotation =  180
                                )
                    temp = controller.last_event.metadata['objects']
                    for t in temp:
                        if t == landmark_name:
                            print(t['Visible'])
                            if t['Visible']:
                                controller.step("Teleport", position = cpos, rotation =  crot
                                )
                                return self.grid2xyz([x,y+i+size],0.91), 180
                    controller.step("Teleport", position = cpos, rotation =  crot
                                )
                    return self.grid2xyz([x,y-i-size],0.91), 0

            if self.grid_map[x+i,y,0] and self.grid_map[x+i+size,y,0]:
                return self.grid2xyz([x+i+size,y],0.91), 270
            elif self.grid_map[x-i,y,0] and self.grid_map[x-i-size,y,0]:
                return self.grid2xyz([x-i-size,y],0.91), 90
            elif self.grid_map[x,y+i,0] and self.grid_map[x,y+i+size,0]:
                return self.grid2xyz([x,y+i+size],0.91), 180
            elif self.grid_map[x,y-i,0] and self.grid_map[x,y-i-size,0]:
                return self.grid2xyz([x,y-i-size],0.91), 0
            else: i+=1

    def plot(self, current_pos, query_object = None):
        cpos = self.xyz2grid(current_pos)
        imshow_grid = copy.deepcopy(self.grid_map)
        imshow_grid[cpos[0],cpos[1],:] = [1,0,0]
        if query_object is not None:
            query_pos = self.xyz2grid(query_object)
            imshow_grid[query_pos[0],query_pos[1],:] = [0,0,1]
        imshow_grid = np.rot90(imshow_grid)
        return imshow_grid

    def xyz2grid(self,pos):
        x = pos['x']
        z = pos['z']
        w = int((x - self.x_min)//self.stepsize)
        h = int((z - self.z_min)//self.stepsize)
        return [w,h]

    def grid2xyz(self,gridmap,y=None):
        if y==None:
            y=self.y_default
        x = gridmap[0] * self.stepsize + self.x_min

        z = gridmap[1] * self.stepsize + self.z_min
        
        return dict(x=x,y=y,z=z)
    
    def setgoalxyz(self,goal):
        self.goal = self.xyz2grid(goal)
    
    def setstartxyz(self,start):
        self.start = self.xyz2grid(start)
        
    def get_rstate(self,reachable_state):
        rstate = []
        for state in reachable_state:
            w,h = self.xyz2grid(state)
            rstate.append([w,h])
            
        self.rstate = rstate