import numpy as np
import matplotlib.pyplot as plt
import copy
class single_scenemap():
    def __init__(self,scenebound, reachable_state):
        self.scenebound = np.asarray(scenebound)
        x_max,y_max = np.max(self.scenebound,axis=0)
        x_min, y_min  = np.min(self.scenebound,axis=0)
        print(x_min,x_max,y_min,y_max)
        self.stepsize = 0.25
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
        self.grid_map = np.ones((x_quan,y_quan))
        print(self.grid_map.shape)
        self.get_gridmap(reachable_state)

    def get_gridmap(self,reachable_state):
        rstate = [[r['x'],r['z']] for r in reachable_state]
        rstate = np.asarray(rstate) # [N x 2]
        rstate[:,0] -= self.x_min
        rstate[:,1] -= self.y_min
        rstate /= 0.25
        rstate = rstate.astype('int32')
        for r in rstate:
            self.grid_map[r[0],r[1]] =0

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



