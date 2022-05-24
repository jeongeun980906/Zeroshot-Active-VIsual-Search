from xml.etree.ElementTree import PI
from ithor_tools.landmark_utils import check_visbility
import numpy as np
import matplotlib.pyplot as plt
import copy
from ithor_tools.landmark_utils import get_gt_box
import math

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
                                stepsize=0.25, margin=0, vis_loi = False):
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
        self.max_obj_size = 20

        self.vis_loi = vis_loi
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
        
    def plot_landmarks(self,controller,show=False):
        self.landmark_loi = []
        self.landmark_loi_name = []
        self.show = show
        for l in self.landmarks:
            pos = self.xyz2grid(l['cp'])
            color = self.landmark_names.index(l['name'])
            lois = self.get_landmark_viewpoint(l['cp'],l['ID'],controller) # World cordinates
            self.landmark_loi.append(lois)
            self.landmark_loi_name.extend([l['name']] *len(lois))
            print(lois)
            for loi in lois:
                loi_grid= self.xyz2grid(loi[0])
                if self.vis_loi:
                    self.grid_map[loi_grid[0],loi_grid[1],:] = self.landmark_colors(color)[:3]

            self.grid_map[pos[0],pos[1],:] = self.landmark_colors(color)[:3]
            

    def get_landmark_viewpoint(self,pos,landmark_ID,controller):
        print(landmark_ID)
        [x,y] = self.xyz2grid(pos)
        pos_ = []
        rot_ = []
        cost_ = []
        min_reachable = self.get_min_reachable_point(x,y)
        for reachable_dict in min_reachable:
            lpos,lrot, step_size = self.get_loi(x,y,reachable_dict,controller,landmark_ID)
            # print(lrot)
            if lpos != None:
                step_size += reachable_dict['len']
                pos_.append(lpos)
                rot_.append(lrot)
                cost_.append(step_size+ 10*(lrot%90 != 0 ))
        if len(cost_) == 0:
            print("Path Not found")
        else:
            temp = copy.deepcopy(cost_)
            temp.sort()
            sorted_pos = []
            sorted_rot = []
            for t in temp:
                min_index = cost_.index(t)
                sorted_pos.append(pos_[min_index])
                sorted_rot.append(rot_[min_index])
            return [[sorted_pos[i],sorted_rot[i]] for i in range(len(temp))]
            # if self.num_loi == 0: # Record all
            #     return [[sorted_pos[i],sorted_rot[i]] for i in range(len(temp))]

            # if len(temp)>1 and self.num_loi == 2:
            #     return [[sorted_pos[i],sorted_rot[i]] for i in range(2)]
           
            # else:
            #     return [[sorted_pos[0],sorted_rot[0]]]

    def check_visibility(self,target_pos,target_rot,controller,landmark_name):
        # cpos = controller.last_event.metadata['agent']['position']
        # crot = controller.last_event.metadata['agent']['rotation']
        try:
            controller.step("Teleport", position = target_pos, rotation =  target_rot
                                        )
        except:
            return False

        if controller.last_event.metadata['lastActionSuccess'] == False:
            return False

        gt_box = get_gt_box(controller,landmark_name,self.show)
        
        if gt_box ==None:
            return False
        box_area = (gt_box[2]-gt_box[0])*(gt_box[3]-gt_box[1])
        frame = controller.last_event.frame
        area = frame.shape[0]*frame.shape[1]
        ratio = box_area/area
        # print(ratio)
        if ratio<0.2 and ratio>0.05 and (gt_box[2]-gt_box[0])>0.7*frame.shape[0]:
            return 0.21
        if ratio<0.2 and ratio>0.05 and gt_box[1]>2:
            return 0.21
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

    def axis2rot(self,axis):
        theta = math.atan2(axis[1],axis[0]) # [-180, 180]
        theta = theta/math.pi*180
        theta = 270-theta
        if theta<0:
            return theta+360
        elif theta>380:
            return theta-360
        else:
            return theta

    def get_min_reachable_point(self,x,y):
        res = []
        total_axis = [[0,1],[1,0],[0,-1],[-1,0],[1/2,1/2],[-1/2,1/2],[-1/2,-1/2],[1/2,-1/2]]
        for axis in total_axis:
            i = self.check_reachable(x,y,axis)
            if i != None:
                res.append(dict(len = i, axis = axis))
        # print(res)
        return res

    def check_reachable(self,x,y,axis):
        for i in range(1,self.max_obj_size):
            step = [int(x+axis[0]*i),int(y+axis[1]*i)] 
            if step[0]<self.grid_map.shape[0]-1 and step[0]>0 and step[1]>0 and step[1]<self.grid_map.shape[1]-1:
                if self.grid_map[step[0],step[1],0]:
                    return i
        return None

    def get_loi(self,x,y,reachable_dict,controller,landmark_ID):
        i = reachable_dict['len']
        axis = reachable_dict['axis']
        pos = [int(x+axis[0]*i),int(y+axis[1]*i)] 
        rot = self.axis2rot(axis)
        size = 10
        max_ratio = 0
        max_size = 0
        while size>0:
            new_pos = [int(pos[0]+axis[0]*size),int(pos[1]+axis[1]*size)]
            if new_pos[0]>0 and new_pos[0]<self.grid_map.shape[0]-1 and new_pos[1]>0 and new_pos[1]<self.grid_map.shape[1]-1:
                if self.grid_map[new_pos[0],new_pos[1],0]:
                    ratio =  self.check_visibility(self.grid2xyz(new_pos,0.91), rot,controller,landmark_ID)
                    # print(ratio)
                    if ratio>0.2 and ratio<0.8:
                        return self.grid2xyz(new_pos,0.91), rot, size
                    
                    # Heuristic return max ratio if nothing is found
                    if max_ratio<ratio:
                        max_size = size
                        max_ratio = ratio
            size -= 2

        if max_ratio>0.1:
            new_pos = [int(pos[0]+axis[0]*max_size),int(pos[1]+axis[1]*max_size)]
            return self.grid2xyz(new_pos,0.91), rot,size
        else:
            return None,None, None