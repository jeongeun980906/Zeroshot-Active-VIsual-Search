import numpy as np
import matplotlib.pyplot as plt
import copy

def giveMargintoGridmap(grid_map,wh_quan,margin_quan):
    ToGiveMargin = []
    print(margin_quan)
    w_quan,h_quan = wh_quan
    for w in range(w_quan):
        for h in range(h_quan):
            if grid_map[w,h] == 1:
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
        x,y = ele
        grid_map[x,y] = 1

    return grid_map

class single_scenemap():
    def __init__(self,scenebound, reachable_state, stepsize=0.25, margin=0):
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
        y_len =  z_max- z_min
        # print(x_min,x_max,z_min,z_max)
        self.x_min = x_min
        self.z_min = z_min
        w_quan = int(x_len//self.stepsize)+1
        h_quan = int(y_len//self.stepsize)+1
        
        self.w_quan = w_quan
        self.h_quan = h_quan
        
        self.grid_map = np.ones((w_quan,h_quan))
        print(self.grid_map.shape)
        self.get_gridmap(reachable_state,margin)

    def get_gridmap(self,reachable_state,margin):
        rstate = [[r['x'],r['z']] for r in reachable_state]
        rstate = np.asarray(rstate) # [N x 2]
        rstate[:,0] -= self.x_min
        rstate[:,1] -= self.z_min
        rstate /= self.stepsize
        rstate = rstate.astype('int32')
        for r in rstate:
            self.grid_map[r[0],r[1]] =0
        
        w_quan = self.w_quan
        h_quan = self.h_quan
        
        # Give margin to gridmap
        margin_quan = int(margin//self.stepsize)
        self.grid_map = giveMargintoGridmap(self.grid_map,(w_quan,h_quan), margin_quan)
        
    def plot(self, current_pos):
        x_pos = int((current_pos['x'] - self.x_min)//self.stepsize)
        z_pos = int((current_pos['z'] - self.z_min)//self.stepsize)
        imshow_grid = copy.deepcopy(self.grid_map)
        imshow_grid[x_pos,z_pos] = 0.5
        imshow_grid = np.rot90(imshow_grid)
        return imshow_grid

    def xyz2grid(self,pos):
        x = pos['x']
        z = pos['z']
        w = int((x - self.x_min)//self.stepsize)
        h = int((z - self.z_min)//self.stepsize)
        return [w,h]

    def grid2xyz(self,gridmap,y):
        x = gridmap[0] * self.stepsize + self.x_min

        z = gridmap[1] * self.stepsize + self.z_min
        
        return dict(x=x,y=y,z=z)
    
    def setgoalxyz(self,goal):
        self.goal = self.xyz2grid(goal)
        