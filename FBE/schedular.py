from ai2thor.util.metrics import (
    get_shortest_path_to_point
)
import numpy as np
import math

class traj_schedular():
    def __init__(self,landmark_names,controller,args=None):
        '''
        buffer: {name, pos, rot} only landmarks
        '''
        self.buffer = [] 
        if args != None:
            if args.scene == 'all':
                self.thres = 2
                if args.fliker:
                    self.ratio = 100
                else:
                    self.ratio = 1
                self.unct_ratio = args.unct_ratio
                
            else:
                self.thres = 0.5 # 1
                self.ratio = 3
                self.unct_ratio = args.unct_ratio
            
            if args.unct_ratio>0:
                self.unct_thres = 2.5 #3
            else:
                self.unct_thres = 100
        else:
            self.thres =  1
            self.ratio = 1
            self.unct_ratio = 0.05
            self.unct_thres = 2.5 

        self.controller = controller
        
        self.landmark_names = landmark_names
        
    def set_score(self,co_occurence):
        self.co_occurence_score = co_occurence

    def schedule(self,cpos,crot, candidate_trajs,uncts,co_thres=0.2):
        register = [dict(name='current',pos=cpos,rot = crot)]
        frontiers = []
        score = [-1]
        unct =[0]
        for c,u in zip(candidate_trajs,uncts):
            if c['name'] != 'frontier':
                flag = True
                for w in self.buffer:
                    ncost = int(w['name'] != c['name'] )
                    if w['rot']==None:
                        rcost = 0
                    else:
                        rcost = (abs(w['rot']- c['rot']))%181
                    dcost = self.get_dis(w['pos'],c['pos'])
                    cost = dcost+ 0.01*rcost+1*ncost # 0.01,1  
                    # print(cost,dcost,rcost,ncost,c['name'],w)
                    if cost<self.thres:
                        flag = False
                if flag:
                    co_score = self.landmark_names.index(c['name'])
                    co_score = self.co_occurence_score[co_score]
                    if co_score> co_thres and u<self.unct_thres:
                        register.append(c)
                        unct.append(u)
                        score.append(co_score) 
            else:
                frontiers.append(c)
        dis_matrix = self.distance(register)
        sorted_indx = self.optimize(score,unct,dis_matrix)
        path = []
        for idx in sorted_indx:
            path.append(register[idx])
        
        min_dis = 100
        last_pos = path[-1]['pos']
        visit_frontier = None
        for f in frontiers:
            # print(f['pos'],last_pos)
            dis = self.shortest_path_length(self.controller,f['pos'],last_pos)
            if dis<min_dis:
                min_dis = dis
                visit_frontier = f
        path.append(visit_frontier)
        self.buffer += register[1:]
        if visit_frontier != None:
            self.buffer += [visit_frontier] # This part out?
        return path

    def distance(self,buffer):
        score_matrix = np.eye(len(buffer))*100
        for i in range(len(buffer)):
            for j in range(len(buffer)):
                if j>=i: 
                    break
                pos1 = buffer[i]['pos']
                pos2 = buffer[j]['pos']
                dis = self.shortest_path_length(self.controller,pos1,pos2)
                score_matrix[i,j] = dis
                score_matrix[j,i] = dis
            
        return score_matrix


    def optimize(self,score, unct,dis_matrix):
        index = [0]
        distance = dis_matrix
        score = self.ratio*(1+1e-3-np.asarray(score))+ self.unct_ratio * np.asarray(unct)
        for i in range(1,len(score)):
            temp = np.arange(len(score))
            temp = np.delete(temp,index)
            dis = distance.copy()[index[-1]] # [N-i x 1]
            dis = np.delete(dis,index) # [N-i-1 x 1]
            scaled_score = np.delete(score,index)
            scaled_score = dis+scaled_score # [N -i -1 x 1]
            new = np.argmin(scaled_score)
            new = temp[new]
            index.append(new)
        return index

    def shortest_path_length(self,controller,goal,init):
        try:
            path = get_shortest_path_to_point(
                    controller=controller,
                        target_position= goal,
                        initial_position= init,
                        allowed_error=0.1
                    )
        except:
            return 100
        distance = 0
        for e in range(len(path)-1):
            dx = abs(path[e]['x'] -  path[e+1]['x'])
            dz = abs(path[e]['z'] - path[e+1]['z'])
            distance += math.sqrt(dx**2+dz**2)
        return distance

    def get_dis(self,x1,x2):
        return math.sqrt((x1['x']-x2['x'])**2+(x1['z']-x2['z'])**2)
