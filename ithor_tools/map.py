import numpy as np
import matplotlib.pyplot as plt
import copy

def giveMargintoGridmap(gridmap,wh_quan,margin_quan):
    ToGiveMargin = []
    print(margin_quan)
    w_quan,h_quan = wh_quan
    for w in range(w_quan):
        for h in range(h_quan):
            if gridmap[w,h] == 1:
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
        gridmap[w,h] = 1

    return gridmap

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
        z_len =  z_max- z_min
        # print(x_min,x_max,z_min,z_max)
        self.x_min, self.x_max = x_min, x_max
        self.z_min, self.z_max = z_min, z_max
        self.y_default = reachable_state[0]['y']
        w_quan = int(x_len//self.stepsize)+1
        h_quan = int(z_len//self.stepsize)+1
        
        self.w_quan = w_quan
        self.h_quan = h_quan
        
        self.gridmap = np.ones((w_quan,h_quan))
        print(self.gridmap.shape)
        self.get_gridmap(reachable_state,margin)
        
        self.rstate = self.get_rstate(reachable_state)
        
    def get_gridmap(self,reachable_state,margin):
        rstate = [[r['x'],r['z']] for r in reachable_state]
        rstate = np.asarray(rstate) # [N x 2]
        rstate[:,0] -= self.x_min
        rstate[:,1] -= self.z_min
        rstate /= self.stepsize
        rstate = rstate.astype('int32')
        for r in rstate:
            self.gridmap[r[0],r[1]] =0
        
        w_quan = self.w_quan
        h_quan = self.h_quan
        
        # Give margin to gridmap
        margin_quan = int(margin//self.stepsize)
        self.gridmap = giveMargintoGridmap(self.gridmap,(w_quan,h_quan), margin_quan)
        
    def plot(self, current_pos):
        x_pos = int((current_pos['x'] - self.x_min)//self.stepsize)
        z_pos = int((current_pos['z'] - self.z_min)//self.stepsize)
        imshow_grid = copy.deepcopy(self.gridmap)
        imshow_grid[x_pos,z_pos] = 0.5
        imshow_grid = np.rot90(imshow_grid)
        return imshow_grid

    def xyz2grid(self,pos):
        x = pos['x']
        z = pos['z']
        w = int((x - self.x_min)//self.stepsize)
        h = int((z - self.z_min)//self.stepsize)
        return [w,h]

    def grid2xyz(self,wh,y=None):
        if y==None:
            y=self.y_default
        x = wh[0] * self.stepsize + self.x_min

        z = wh[1] * self.stepsize + self.z_min
        
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
            
        return  rstate