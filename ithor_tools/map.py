import numpy as np
import matplotlib.pyplot as plt
import copy

def giveMargintoGridmap(grid_map,xy_quan,margin_quan):
    ToGiveMargin = []
    print(margin_quan)
    x_quan,y_quan = xy_quan
    for x in range(x_quan):
        for y in range(y_quan):
            if grid_map[x,y] == 1:
                # if (x,y) is occupied
                
                # make it's neighbor occuppied with with margin
                for x_margin in range(2*margin_quan+1):
                    for y_margin in range(2*margin_quan+1):
                        x_neighbor = x + x_margin - margin_quan
                        y_neighbor = y + y_margin - margin_quan
                        if 0<=x_neighbor and x_neighbor<x_quan and 0<=y_neighbor and y_neighbor<y_quan:
                            ToGiveMargin.append((x_neighbor,y_neighbor))
                        
    ToGiveMargin = set(ToGiveMargin)
    ToGiveMargin = [list(ele) for ele in ToGiveMargin]
    
    for ele in ToGiveMargin:
        x,y = ele
        grid_map[x,y] = 1

    return grid_map

class single_scenemap():
    def __init__(self,scenebound, reachable_state, stepsize=0.25, margin=0):
        self.scenebound = np.asarray(scenebound)
        x_max,y_max = np.max(self.scenebound,axis=0)
        x_min, y_min  = np.min(self.scenebound,axis=0)
        print(x_min,x_max,y_min,y_max)
        self.stepsize = stepsize
        x_max = self.stepsize* (x_max//self.stepsize)
        y_max = self.stepsize* (y_max//self.stepsize)
        x_min = self.stepsize* (x_min//self.stepsize +1)
        y_min = self.stepsize* (y_min//self.stepsize +1)

        x_len =  x_max- x_min
        y_len =  y_max- y_min
        print(x_min,x_max,y_min,y_max)
        self.x_min = x_min
        self.y_min = y_min
        x_quan = int(x_len//self.stepsize)+1
        y_quan = int(y_len//self.stepsize)+1
        
        self.x_quan = x_quan
        self.y_quan = y_quan
        
        self.grid_map = np.ones((x_quan,y_quan))
        print(self.grid_map.shape)
        self.get_gridmap(reachable_state,margin)

    def get_gridmap(self,reachable_state,margin):
        rstate = [[r['x'],r['z']] for r in reachable_state]
        rstate = np.asarray(rstate) # [N x 2]
        rstate[:,0] -= self.x_min
        rstate[:,1] -= self.y_min
        rstate /= self.stepsize
        rstate = rstate.astype('int32')
        for r in rstate:
            self.grid_map[r[0],r[1]] =0
        
        x_quan = self.x_quan
        y_quan = self.y_quan
        
        # Give margin to gridmap
        margin_quan = int(margin//self.stepsize)
        self.grid_map = giveMargintoGridmap(self.grid_map,(x_quan,y_quan), margin_quan)
        
        
                    
            
        

    def plot(self, current_pos):
        x_pos = int((current_pos[0] - self.x_min)//self.stepsize)
        y_pos = int((current_pos[1] - self.y_min)//self.stepsize)
        imshow_grid = copy.deepcopy(self.grid_map)
        imshow_grid[x_pos,y_pos] = 0.5
        imshow_grid = np.rot90(imshow_grid)
        return imshow_grid

    def gridmap_pos(self,pos):
        x_pos = int((pos[0] - self.x_min)//self.stepsize)
        y_pos = int((pos[1] - self.y_min)//self.stepsize)
        return [x_pos,y_pos]

    def gridmap2xyz(self,gridmap,y):
        x = gridmap[0] * self.stepsize + self.x_min

        z = gridmap[1] * self.stepsize + self.y_min
        
        return dict(x=x,y=y,z=z)