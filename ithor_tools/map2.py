from ithor_tools.landmark_utils import check_visbility
import numpy as np
import matplotlib.pyplot as plt
import copy
from ithor_tools.landmark_utils import get_gt_box
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
        
    def plot_landmarks(self,controller):
        self.landmark_loi = []
        for l in self.landmarks:
            pos = self.xyz2grid(l['cp'])
            color = self.landmark_names.index(l['name'])
            loi = self.get_landmark_viewpoint(l['cp'],l['ID'],controller) # World cordinates
            self.landmark_loi.append(loi)
            loi_grid = self.xyz2grid(loi[0])
            self.grid_map[pos[0],pos[1],:] = self.landmark_colors(color)[:3]
            self.grid_map[loi_grid[0],loi_grid[1],:] = self.landmark_colors(color)[:3]

    def get_landmark_viewpoint(self,pos,landmark_ID,controller):
        print(landmark_ID)
        [x,y] = self.xyz2grid(pos)
        i = 1
        while i<20:
            lpos,lrot = self.save_len(x,y,i,controller,landmark_ID)
            if lrot == None:
                lpos,lrot = self.axis_algin(x,y,i,controller,landmark_ID)
                if lrot is not None:
                    return lpos,lrot
            else:
                return lpos,lrot
            i+=1
            # ratio /= 1.2
        # ratio = 50
        # while ratio>20:
        #     size = int(self.stepsize*ratio)
        i = 1
        while i<20:
            lpos,lrot = self.save_len(x,y,i,controller,landmark_ID)
            if lrot == None:
                lpos,lrot = self.axis_algin(x,y,i,controller,landmark_ID)
                if lrot is not None:
                    return lpos,lrot
            else: 
                return lpos,lrot
            if lrot == None:
                lpos,lrot = self.diagonal(x,y,i,controller,landmark_ID)
                if lrot is not None:
                    return lpos,lrot
            else:
                return lpos,lrot
            i+=1
        #     ratio /= 1.2
        print("Path Not found")
        # return lrot,lrot

    def check_visibility(self,target_pos,target_rot,controller,landmark_name):
        cpos = controller.last_event.metadata['agent']['position']
        crot = controller.last_event.metadata['agent']['rotation']
        try:
            controller.step("Teleport", position = target_pos, rotation =  target_rot
                                        )
        except:
            return 0
        gt_box = get_gt_box(controller,landmark_name)
        
        if gt_box ==None:
            return False
        box_area = (gt_box[2]-gt_box[0])*(gt_box[3]-gt_box[1])
        frame = controller.last_event.frame
        area = frame.shape[0]*frame.shape[1]
        ratio = box_area/area
        print(ratio)
        # controller.step("Teleport", position = cpos, rotation =  crot
        #             )
        # temp = controller.last_event.metadata['objects']
        # for t in temp:
        #     if t['objectId'] == landmark_ID:
        #         # print(t['visible'])
        #         if t['visible']:
        #             controller.step("Teleport", position = cpos, rotation =  crot
        #             )
        #             return True
        # controller.step("Teleport", position = cpos, rotation =  crot
        #             )
        return ratio
        # if ratio<0.8 and ratio>0.2:
        #     return 1
        # elif ratio>0:
        #     return 0.5
        # else:
        #     return 0

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

    def save_len(self,x,y,i,controller,landmark_ID):
        if x+i<self.grid_map.shape[0]-1 and x-i>0:
            if self.grid_map[x+i,y,0]*self.grid_map[x-i,y,0]:
                size = 100
                while size>3:
                    if x+i+size<self.grid_map.shape[0]-1:
                        if self.grid_map[x+i+size,y,0]:
                            ratio =  self.check_visibility(self.grid2xyz([x+i+size,y],0.91), 270,controller,landmark_ID)
                            if ratio>0.2 and ratio<0.8:
                                return self.grid2xyz([x+i+size,y],0.91), 270
                    if x-i-size>0:
                        if self.grid_map[x-i-size,y,0]: 
                            ratio =  self.check_visibility(self.grid2xyz([x-i-size,y],0.91), 90,controller,landmark_ID)
                            if ratio>0.2 and ratio<0.8:
                                return self.grid2xyz([x-i-size,y],0.91), 90
                    size -=4

        if y+i<self.grid_map.shape[1]-1 and y-i>0:
            if self.grid_map[x,y+i,0]*self.grid_map[x,y-i,0]:
                size = 100
                while size>3:
                    if y+i+size<self.grid_map.shape[1]-1:
                        if self.grid_map[x,y+i+size,0]:
                            ratio =  self.check_visibility(self.grid2xyz([x,y+i+size],0.91),180,controller,landmark_ID)
                            if ratio>0.2 and ratio<0.8:
                                return self.grid2xyz([x,y+i+size],0.91), 180
                    if y-i-size>0:
                        if self.grid_map[x,y-i-size,0]:
                            ratio = self.check_visibility(self.grid2xyz([x,y-i-size],0.91),0,controller,landmark_ID)
                            if ratio>0.2 and ratio<0.8:
                                return self.grid2xyz([x,y-i-size],0.91), 0
                    size -=4

        return False,None

    def diagonal(self,x,y,i,controller,landmark_ID):
        if x-int(i/2)>0 and y-int(i/2)>0:
            if self.grid_map[x-int(i/2),y-int(i/2),0]:
                size = 100
                prev_ratio = 0
                downflag = 0
                upflag = 0
                while size>3:
                    if x-int(i/2+size/2)>0 and y-int(i/2+size/2)>0:
                        if self.grid_map[x-int(i/2+size/2),y-int(i/2+size/2),0]:
                            ratio =  self.check_visibility(self.grid2xyz([x-int(i/2+size/2),y-int(i/2+size/2)],0.91), 45,controller,landmark_ID)
                            if ratio>0.2 and ratio<0.8:
                                return self.grid2xyz([x-int(i/2+size/2),y-int(i/2+size/2)],0.91), 45
                            elif ratio>prev_ratio:
                                upflag += 1
                            elif ratio<prev_ratio:
                                downflag +=1

                            if downflag>0 and upflag>0:
                                size += 4
                                return self.grid2xyz([x-int(i/2+size/2),y-int(i/2+size/2)],0.91), 45

                            prev_ratio = ratio
                                
                    size -=4

        if x+int(i/2)<self.grid_map.shape[0]-1  and y-int(i/2)>0:
            if self.grid_map[x+int(i/2),y-int(i/2),0]:
                size = 100
                prev_ratio = 0
                downflag = 0
                upflag = 0
                while size>3:
                    if x+int(i/2+size/2)<self.grid_map.shape[0]-1  and y-int(i/2+size/2)>0:
                        if self.grid_map[x+int(i/2+size/2),y-int(i/2+size/2),0]:
                            ratio =  self.check_visibility(self.grid2xyz([x+int(i/2+size/2),y-int(i/2+size/2)],0.91), 315,controller,landmark_ID)
                            if ratio>0.2 and ratio<0.8:
                                return self.grid2xyz([x+int(i/2+size/2),y-int(i/2+size/2)],0.91), 315
                            elif ratio>prev_ratio:
                                upflag += 1
                            elif ratio<prev_ratio:
                                downflag +=1

                            if downflag>0 and upflag>0:
                                size += 4
                                return self.grid2xyz([x+int(i/2+size/2),y-int(i/2+size/2)],0.91), 315

                            prev_ratio = ratio
                                
                    size -=4

        if x+int(i/2)< self.grid_map.shape[0]-1 and y+int(i/2)<self.grid_map.shape[1]-1:
            if self.grid_map[x+int(i/2),y+int(i/2),0]:
                size = 100
                prev_ratio = 0
                downflag = 0
                upflag = 0
                while size>3:
                    if x+int(i/2+size/2)< self.grid_map.shape[0]-1 and y+int(i/2+size/2)<self.grid_map.shape[1]-1:
                        if self.grid_map[x+int(i/2+size/2),y+int(i/2+size/2),0]:
                            ratio =  self.check_visibility(self.grid2xyz([x+int(i/2+size/2),y+int(i/2+size/2)],0.91), 225 ,controller,landmark_ID)
                            if ratio>0.2 and ratio<0.8:
                                return self.grid2xyz([x+int(i/2+size/2),y+int(i/2+size/2)],0.91), 225
                            elif ratio>prev_ratio:
                                upflag += 1
                            elif ratio<prev_ratio:
                                downflag +=1

                            if downflag>0 and upflag>0:
                                size += 4
                                return self.grid2xyz([x+int(i/2+size/2),y+int(i/2+size/2)],0.91), 225

                            prev_ratio = ratio
                                
                                
                    size -=4
                size = 50
          
        if x-int(i/2)>0 and y+int(i/2)<self.grid_map.shape[1]-1 :
            if self.grid_map[x-int(i/2),y+int(i/2),0]:
                size = 100
                prev_ratio = 0
                downflag = 0
                upflag = 0
                while size>3:
                    if x-int(i/2+size/2)>0 and y+int(i/2+size/2)<self.grid_map.shape[1]-1:
                        if self.grid_map[x-int(i/2+size/2),y+int(i/2+size/2),0]:
                            ratio = self.check_visibility(self.grid2xyz([x-int(i/2+size/2),y+int(i/2+size/2)],0.91), 135,controller,landmark_ID)
                            if ratio>0.2 and ratio<0.8:
                                return self.grid2xyz([x-int(i/2+size/2),y+int(i/2+size/2)],0.91), 135
                            elif ratio>prev_ratio:
                                upflag += 1
                            elif ratio<prev_ratio:
                                downflag +=1

                            if downflag>0 and upflag>0:
                                size += 4
                                return self.grid2xyz([x-int(i/2+size/2),y+int(i/2+size/2)],0.91), 135

                            prev_ratio = ratio
                                
                    size -=4
        return False,None

    def axis_algin(self,x,y,i,controller,landmark_ID):
        if x+i<self.grid_map.shape[0]-1:
            if self.grid_map[x+i,y,0]:
                size = 100
                prev_ratio = 0
                downflag = 0
                upflag = 0
                while size>3:
                    if x+i+size<self.grid_map.shape[0]-1:
                        if self.grid_map[x+i+size,y,0]:
                            ratio =  self.check_visibility(self.grid2xyz([x+i+size,y],0.91), 270,controller,landmark_ID)
                            if ratio>0.2 and ratio<0.8:
                                return self.grid2xyz([x+i+size,y],0.91), 270
                            elif ratio>prev_ratio:
                                upflag += 1
                            elif ratio<prev_ratio:
                                downflag +=1

                            if downflag>0 and upflag>0:
                                size += 4
                                return self.grid2xyz([x+i+size,y],0.91), 270

                            prev_ratio = ratio
                    size -=4
                
        if x-i>0:
            if self.grid_map[x-i,y,0]:
                size = 100
                prev_ratio = 0
                downflag = 0
                upflag = 0
                while size>3:
                    if x-i-size>0:
                        if self.grid_map[x-i-size,y,0]:
                            ratio = self.check_visibility(self.grid2xyz([x-i-size,y],0.91), 90,controller,landmark_ID)
                            if ratio>0.2 and ratio<0.8:
                                return self.grid2xyz([x-i-size,y],0.91), 90
                            elif ratio>prev_ratio:
                                upflag += 1
                            elif ratio<prev_ratio:
                                downflag +=1

                            if downflag>0 and upflag>0:
                                size += 4
                                return self.grid2xyz([x-i-size,y],0.91), 90
                            prev_ratio = ratio
                    size-=4


        if y+i<self.grid_map.shape[1]-1:
            if self.grid_map[x,y+i,0]:
                size = 100
                prev_ratio = 0
                downflag = 0
                upflag = 0
                while size>3:
                    if y+i+size<self.grid_map.shape[1]-1:
                        if self.grid_map[x,y+i+size,0]:
                            ratio =  self.check_visibility(self.grid2xyz([x,y+size+i],0.91), 180,controller,landmark_ID)
                            if ratio>0.2 and ratio<0.8:
                                return self.grid2xyz([x,y+i+size],0.91), 180
                            elif ratio>prev_ratio:
                                upflag += 1
                            elif ratio<prev_ratio:
                                downflag +=1

                            if downflag>0 and upflag>0:
                                size += 4
                                return self.grid2xyz([x,y+i+size],0.91), 180
                            prev_ratio = ratio
                    size-=4
                        
        if y-i>0:
            if self.grid_map[x,y-i,0]:
                size = 100
                prev_ratio = 0
                downflag = 0
                upflag = 0
                while size>3:
                    if y-i-size>0:
                        if self.grid_map[x,y-i-size,0]:
                            ratio =  self.check_visibility(self.grid2xyz([x,y-i-size],0.91), 0,controller,landmark_ID)
                            if ratio>0.2 and ratio<0.8:
                                return self.grid2xyz([x,y-i-size],0.91), 0
                            elif ratio>prev_ratio:
                                upflag += 1
                            elif ratio<prev_ratio:
                                downflag +=1

                            if downflag>0 and upflag>0:
                                size += 4
                                return self.grid2xyz([x,y-i-size],0.91), 0
                            prev_ratio = ratio
                    size-=4

        return False,None