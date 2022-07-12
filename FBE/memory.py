import open3d as o3d
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
from FBE.union import UnionFind
import copy

def to_rad(th):
    return th*math.pi / 180


class fbe_map():
    def __init__(self,scenebound, reachable_state, landmark_names, 
                                stepsize=0.25, margin=0, vis_loi = False):
        scenebound = np.asarray(scenebound)
        x_max, z_max = np.max(scenebound,axis=0)
        x_min, z_min  = np.min(scenebound,axis=0)
        # print(x_min,x_max,z_min,z_max)
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
        self.ray_length = 1.7
        self.vis_length = self.ray_length/stepsize

        self.vis_loi = vis_loi
        self.w_quan = w_quan
        self.h_quan = h_quan
        
        self.gt_grid_map = np.zeros((w_quan,h_quan,3))
        self.grid_map = np.ones((w_quan,h_quan,3))/2

        self.landmark_names = landmark_names

        self.landmark_colors = plt.cm.get_cmap('tab20b', len(landmark_names))
        self.get_gridmap(reachable_state,margin)
        
        self.get_rstate(reachable_state)
        self.robot_size = 3
        width = 800
        height = 800
        fov = 60
        self.width = width
        self.height = height
        # camera intrinsics
        focal_length = 0.5 * width / math.tan(to_rad(fov/2))
        fx, fy, cx, cy = (focal_length,focal_length, width/2, height/2)
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, 
                                fx, fy, cx, cy)
    
    def reset_gridmap(self):
        self.grid_map = np.ones((self.w_quan,self.h_quan,3))/2

    def get_gridmap(self,reachable_state,margin):
        rstate = [[r['x'],r['z']] for r in reachable_state]
        rstate = np.asarray(rstate) # [N x 2]
        rstate[:,0] -= self.x_min
        rstate[:,1] -= self.z_min
        rstate /= self.stepsize
        rstate = rstate.astype('int32')
        for r in rstate:
            self.gt_grid_map[r[0],r[1],:] =[1,1,1]
        self.unct_map = np.zeros((self.w_quan,self.h_quan))
            
    def visited(self,controller):
        cpos = controller.last_event.metadata['agent']['position']
        cgrid = self.xyz2grid(cpos)
        self.grid_map[cgrid[0],cgrid[1]] = [1,1,1]
        for i in np.arange(start=-1,stop=1.01,step=0.01):
            self.ray_x(cgrid,i) 
            self.ray_y(cgrid,i) 


    def proj(self,bboxs,classes,uncts,controller,visible_landmark_names):
        boxfill = np.zeros((800,800,3)).astype(np.uint8)
        agent_pos = controller.last_event.metadata['agent']['position']
        agent_rot  = controller.last_event.metadata['agent']['rotation']['y']
        frame = controller.last_event.frame
        size = 0.2
        saved_color = []
        saved_name = []
        saved_unct = []
        for box, lname,unct in zip(bboxs,classes,uncts):
            height = box[3] - box[1]
            width = box[2] - box[0]
            center_x = (box[3]+box[1])/2
            center_y = (box[2]+box[0])/2
            area = width*height
            if area> 1e-2*(800*800):
                n_x_l = int(center_x - size*height)
                n_x_u = int(center_x + size*height)
                n_y_l = int(center_y - size*width)
                n_y_u = int(center_y + size*width)
                lname_index = lname.item() #visible_landmark_names.index(lname)
                lname = visible_landmark_names[lname_index]
                color = self.landmark_colors(lname_index)
                color = [int(255*color[0]),int(255*color[1]),int(255*color[2])]
                # print(color)
                boxfill[n_x_l:n_x_u,n_y_l:n_y_u] = color
                saved_color.append(color)
                saved_unct.append(unct)
                saved_name.append(lname)

        res,vpos = self.get_projection(controller,boxfill,agent_pos,agent_rot)
        # plt.figure()
        # plt.subplot(1,2,1)
        # plt.imshow(res)
        # plt.subplot(1,2,2)
        # plt.imshow(boxfill)
        # plt.show()
        detected_landmarks = []
        for point in vpos:
            pos = point['pos']
            color = (point['color']*255).astype(np.int16)
            color = tuple(color)
            ccolor = None
            for c in saved_color:
                if abs(c[0]-color[0])+abs(c[1]-color[1])+abs(c[2]-color[2])<4:
                    ccolor = c
                    break
            # print(color)
            if ccolor != None:
                class_name = saved_name[saved_color.index(ccolor)]
                pos = dict(x=pos[2],y=pos[1],z=pos[0])
                new_color = self.landmark_names.index(class_name)
                new_color = list(self.landmark_colors(new_color))
                map_pos = self.xyz2grid(pos)
                # print(class_name,map_pos)
                if (self.grid_map[map_pos[0],map_pos[1]] == [0,0,0]).all() or (self.grid_map[map_pos[0],map_pos[1]] == [0.5,0.5,0.5]).all():
                    
                    self.grid_map[map_pos[0],map_pos[1]] = new_color[:3]
                    self.unct_map[map_pos[0],map_pos[1]] = saved_unct[saved_color.index(ccolor)]
                    detected_landmarks.append(class_name)
        
        return detected_landmarks
            
    def get_projection(self,controller,COLOR,agent_pos,rot):
        DEPTH = controller.last_event.depth_frame
        # GRAY = np.mean(COLOR,axis=-1)/255
        # GRAY = GRAY.astype(np.float32)

        depth = o3d.geometry.Image(DEPTH)
        color = o3d.geometry.Image(COLOR)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth,
                                                                    depth_scale=1.0,
                                                                    depth_trunc=2.0,
                                                                    convert_rgb_to_intensity=False)
        # o3d.visualization.draw_geometries([rgbd])
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.intrinsic)
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        
        rot = math.pi*rot/180
        pcd.transform([[-math.sin(rot), 0,-math.cos(rot), agent_pos['z']],
                [0, 1, 0, agent_pos['y']],
                [math.cos(rot), 0, -math.sin(rot), agent_pos['x']],
                [0, 0, 0, 1]])

        # o3d.visualization.draw_geometries([pcd])
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,
                                                                    voxel_size=0.1)
        # o3d.visualization.draw_geometries([voxel_grid])
        voxels = voxel_grid.get_voxels()  # returns list of voxels
        indices = np.stack(list(vx.grid_index for vx in voxels))
        shape = indices.max(axis=0)
        res = np.zeros((shape[2]+1,shape[0]+1,3))
        vpos= []
        for vx in voxels:
            grid_index = vx.grid_index
            pos_temp = voxel_grid.get_voxel_center_coordinate(grid_index)
            if pos_temp[1]<1.7: # Reject
                color = vx.color
                res[grid_index[2],grid_index[0]] = color
                if round(color[0],3) > 0 :
                    vpos.append(dict(pos=pos_temp,color = color))
                elif pos_temp[1]>0.1:
                    map_pos = dict(x=pos_temp[2],y=pos_temp[1],z=pos_temp[0])
                    map_pos = self.xyz2grid(map_pos)
                    try:
                        self.grid_map[map_pos[0],map_pos[1],:] = [0,0,0]
                    except:
                        pass
        del pcd, voxel_grid
        return res,vpos

    def ray_x(self,start_pos,angle): # -1~1
        x_ = np.arange(start_pos[0]+1,start_pos[0]+self.ray_length/self.stepsize).astype(np.int16)
        y_ = ((angle*(x_-start_pos[0]))+start_pos[1]).astype(np.int16)
        # if sum(temp_map[x_[0],y_[0]+sign[0]:y_[0]+sign[1],0]==0)==0:
        for x,y in zip(x_,y_): #or sum(self.grid_map[x+1:x+2,y,0] !=1)>0:
            if x>0 and x<self.grid_map.shape[0]-1 and y>0 and y<self.grid_map.shape[1]-1: 
                if (self.grid_map[x,y] ==0.5).all() or (self.grid_map[x,y] == 1).all():
                    self.grid_map[x,y] = [1,1,1]
                else:
                    break
                
        x_ = np.arange(start_pos[0]-1,start_pos[0]-self.ray_length/self.stepsize,-1).astype(np.int16)
        y_ = ((angle*(x_-start_pos[0]))+start_pos[1]).astype(np.int16)
        # if  sum(self.grid_map[x_[0],y_[0]+1:y_[0]+2,0]!=1)==0:
        if x>0 and x<self.grid_map.shape[0]-1 and y>0 and y<self.grid_map.shape[1]-1: 
            for x,y in zip(x_,y_)  or sum(self.grid_map[x-1:x,y,0]!=1)>0:
                if (self.grid_map[x,y]==0.5).all() or (self.grid_map[x,y] == 1).all():
                    self.grid_map[x,y] = [1,1,1]
                else:
                    break

    def ray_y(self,start_pos,angle): # -1~1
        y_ = np.arange(start_pos[1]+1,start_pos[1]+self.ray_length/self.stepsize).astype(np.int16)
        x_ = ((angle*(y_-start_pos[1]))+start_pos[0]).astype(np.int16)
        # if  sum(self.grid_map[x_[0]+1:x_[0]+2,y_[0],0]!=1)==0:
        for x,y in zip(x_,y_):
            if x>0 and x<self.grid_map.shape[0]-1 and y>0 and y<self.grid_map.shape[1]-1: 
                if (self.grid_map[x,y] ==0.5).all() or (self.grid_map[x,y] == 1).all(): # or sum(self.grid_map[x,y-1:y,0] !=1)>0: #(temp_map[x,y-1,0]==0 and sum(temp_map[x-1:x+3,y,0]))>1:
                    self.grid_map[x,y] = [1,1,1]
                else:
                    break
        y_ = np.arange(start_pos[1]-1,start_pos[1]-self.ray_length/self.stepsize,-1).astype(np.int16)
        x_ = ((angle*(y_-start_pos[1]))+start_pos[0]).astype(np.int16)
        for x,y in zip(x_,y_):
            if x>0 and x<self.grid_map.shape[0]-1 and y>0 and y<self.grid_map.shape[1]-1: 
                if (self.grid_map[x,y] ==0.5).all() or (self.grid_map[x,y] ==1).all(): #or sum(self.grid_map[x,y+1:y+2,0]!=1)>0:
                    self.grid_map[x,y] = [1,1,1]
                else:
                    break


    def plot(self, current_pos, candidate_trajs=None,query_object = None):
        imshow_grid = copy.deepcopy(self.grid_map)
        for traj in candidate_trajs:
            if traj is not None:
                if traj['name'] == 'frontier':
                    new_color = [1,0.5,0,0]
                else:
                    new_color = self.landmark_names.index(traj['name'])
                    new_color = list(self.landmark_colors(new_color))
                query_pos = self.xyz2grid(traj['pos'])
                imshow_grid[query_pos[0],query_pos[1],:] = new_color[:3]
        cpos = self.xyz2grid(current_pos)
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

    def get_reachable(self,cpos,landmark_name):
        # plt.imshow(self.grid_map)
        # plt.show()
        index = self.landmark_names.index(landmark_name)
        color = self.landmark_colors(index)
        R = (self.grid_map[:,:,0]==color[0])
        G = (self.grid_map[:,:,1]==color[1])
        B = (self.grid_map[:,:,2]==color[2])
        total = R & G & B
        grid_poss = np.where(total>0)
        uncts = self.unct_map[total[:,:]]
        mean_unct = np.mean(uncts)
        # print(mean_unct)
        try:
            x = int(np.mean(grid_poss[0]))
            y = int(np.mean(grid_poss[1]))
        except:
            return None,None,None
        axiss = [[0,1],[1,0],[0,-1],[-1,0],
                [1/2,1/2],[-1/2,1/2],[-1/2,-1/2],[1/2,-1/2]]
        step_size = 5
        min_dis = 100
        res = None
        rpos = None
        raxis = None
        for axis in axiss:
            offset = self.check_reachable(x,y,axis,self.max_obj_size)
            # print(offset)
            if offset != None:
                new_x = int(x+axis[0]*offset)
                new_y = int(y+axis[1]*offset)
                # self.grid_map[new_x,new_y] = [0.5,1,0]
                new_offest = self.check_reachable2(new_x,new_y,axis,step_size)
                if new_offest != None:
                    res_x = int(x+axis[0]*(offset+new_offest))
                    res_y = int(y+axis[1]*(offset+new_offest))
                    rpos = self.grid2xyz([res_x,res_y])
                    dis = self.get_dis(rpos,cpos)
                    if dis<min_dis:
                        res = rpos
                        min_dis = dis
                        raxis = axis
        # if rpos != None:
        #     rgrid = self.xyz2grid(rpos)
        #     self.grid_map[rgrid[0],rgrid[1]] = [1,0,1]
        if res != None:
            return res,self.axis2rot(raxis),min_dis,mean_unct
        else:
            return None,None,None,None

    def reset_landmark(self,landmark_names):
        for landmark_name in landmark_names:
            index = self.landmark_names.index(landmark_name)
            color = self.landmark_colors(index)
            R = (self.grid_map[:,:,0]==color[0])
            G = (self.grid_map[:,:,1]==color[1])
            B = (self.grid_map[:,:,2]==color[2])
            total = R & G & B
            self.grid_map[total] = [0,0,0]
        self.unct_map = np.zeros((self.w_quan,self.h_quan))

    def check_reachable(self,x,y,axis,step_size):
        step = range(1,step_size)
        for i in step:
            x_range = list(range(x,int(x+axis[0]*i)+1)) if axis[0]>0 else list(range(int(x+axis[0]*i),x+1))
            y_range = list(range(y,int(y+axis[1]*i)+1)) if axis[1]>0 else list(range(int(y+axis[1]*i),y+1))
            new_pos = [int(x+axis[0]*(i+1)),int(y+axis[1]*(i+1))]
            if len(x_range) != len(y_range) and (len(x_range)>1 and len(y_range)>1):
                # print(len(x_range),len(y_range))
                if len(x_range)>len(y_range):
                    y_range.append(y_range[-1])
                else:
                    x_range.append(x_range[-1])
                # print(len(x_range),len(y_range))
            if x_range[-1]<self.grid_map.shape[0]-2 and x_range[0]>1 and y_range[0]>1 and y_range[-1]<self.grid_map.shape[1]-2:
                # print(sum(self.grid_map[x_range,y_range] == [1,1,1]))
                if sum(self.grid_map[x_range,y_range] == [1,1,1]).all() >0: #  and sum(self.grid_map[x_range,y_range] == [0.5,0.5,0.5]).all() ==0 and self.grid_map[new_pos[0],new_pos[1],0] == 1:
                    return i+1
        return None
    
    def check_reachable2(self,x,y, axis,step_size):
        for i in range(step_size,1,-1):
            x_range = list(range(x,int(x+axis[0]*i)+1)) if axis[0]>0 else list(range(int(x+axis[0]*i),x+1))
            y_range = list(range(y,int(y+axis[1]*i)+1)) if axis[1]>0 else list(range(int(y+axis[1]*i),y+1))
            if len(x_range) != len(y_range) and (len(x_range)>1 and len(y_range)>1):
                # print(len(x_range),len(y_range))
                if len(x_range)>len(y_range):
                    y_range.append(y_range[-1])
                else:
                    x_range.append(x_range[-1])
            if x_range[-1]<self.grid_map.shape[0]-1 and x_range[0]>0 and y_range[0]>0 and y_range[-1]<self.grid_map.shape[1]-1:
                # print(sum(self.grid_map[x_range,y_range] != [1,1,1]).all())
                if sum(self.grid_map[x_range,y_range] != [1,1,1]).all() ==0:
                    return i
        return None

    def frontier_detection(self,cpos):
        img_gray_recolor = np.zeros((self.grid_map.shape[0],self.grid_map.shape[1]))
        R = (self.grid_map[:,:,0]==1)
        G = (self.grid_map[:,:,1]==1)
        B = (self.grid_map[:,:,2]==1)
        total = R & G & B
        img_gray_recolor[total] = 1
        img_gray_recolor = (img_gray_recolor*255).astype(np.uint8)
        
        edges = cv2.Canny(img_gray_recolor,20,10)
        index = np.where(edges != 0)
        
        frontier_map = np.zeros_like(img_gray_recolor)
        res = []
        for indx in zip(index[0],index[1]):
            if indx[0]+1<self.grid_map.shape[0]:
                right = int(self.grid_map[indx[0]+1,indx[1],0]==0.5)
            else:
                right = 0
            if indx[0]>0:
                left = int(self.grid_map[indx[0]-1,indx[1],0]==0.5)
            else:
                left= 0
            if indx[1]+1 < self.grid_map.shape[1]:
                up = int(self.grid_map[indx[0],indx[1]+1,0]==0.5)
            else:
                up = 0
            if indx[1]>0:
                down = int(self.grid_map[indx[0],indx[1]-1,0] == 0.5)
            else:
                down = 0
            center = int(self.grid_map[indx[0],indx[1],0]==0.5)
            if self.grid_map[indx[0],indx[1],0] != 0 and self.grid_map[indx[0],indx[1],1] != 0 and self.grid_map[indx[0],indx[1],2] != 0: 
                if left+right+up+down > 0 or center :
                    frontier_map[indx[0],indx[1]]=1
                    res.append(indx)
                
        plt.imshow(frontier_map)
        plt.show()
        groups = self.groupTPL(res)
        # filter_by_size = []
        # distances = []
        res = []
        for group in groups:
            if len(group)>self.robot_size:
                mean_x = sum([x[0] for x in group])/len(group)
                mean_y = sum([y[1] for y in group])/len(group)
                frontier_map[int(mean_x),int(mean_y)] = 0.5
                mean = [int(mean_x),int(mean_y)]
                mean = self.grid2xyz(mean)
                res.append(dict(name = 'frontier',pos = mean, rot = None))
        #         try:
        #             path = get_shortest_path_to_point(self.controller,cpos,mean)
        #             dis = 0
        #             last_pos = cpos
        #             for p in path:
        #                 dis += math.sqrt((last_pos['x']-p['x'])**2+(last_pos['z']-p['z'])**2)
        #                 last_pos = p
        #         except:
        #             dis = 100
                
        #         distances.append(dis)
        #         filter_by_size.append(mean)
        # del groups,img_gray_recolor,img_gray,edges
        # if len(distances)>0:
        #     sort_index = np.argsort(np.asarray(distances))
        #     return filter_by_size,sort_index
        # else:
        #     return [],None
        return res
        
    def get_dis(self,x1,x2):
        return math.sqrt((x1['x']-x2['x'])**2+(x1['z']-x2['z'])**2)

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
    def groupTPL(self,TPL, distance=1):
        U = UnionFind()

        for (i, x) in enumerate(TPL):
            for j in range(i + 1, len(TPL)):
                y = TPL[j]
                if max(abs(x[0] - y[0]), abs(x[1] - y[1])) <= distance:
                    U.union(x, y)

        disjSets = {}
        for x in TPL:
            s = disjSets.get(U[x], set())
            s.add(x)
            disjSets[U[x]] = s

        return [list(x) for x in disjSets.values()]