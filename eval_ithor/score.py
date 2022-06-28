import pandas as pd
import json
import os
class score_storage():
    def __init__(self,args):
        self.buffer = []
        if args.version != None:
            if args.val:
                self.file_path = './res/v%d_val_%d_%d'%(args.version,args.base_detector,args.dis_only)
            else:
                self.file_path = './res/v%d_%d_%d'%(args.version,args.base_detector,args.dis_only)
        else:
            if args.co_base:
                num = 1
            elif args.dis_only:
                num =2 
            else :
                num=0
            if args.val:
                self.file_path = './res/val_%d_%d_%d'%(args.base_detector,num,args.num_loi)
            else:
                self.file_path = './res/%d_%d_%d'%(args.base_detector,num,args.num_loi)
        try:
            os.mkdir(self.file_path)
        except:
            pass
    def append(self,score,query_object_name,scene_name):
        self.buffer.append(dict(score=score,query_name=query_object_name,
                        scene_name=scene_name))
        
    def save_json(self):
        with open("{}/out.json".format(self.file_path),"w") as jf:
            json.dump(self.buffer,jf,indent=4)

    def average(self):
        df = pd.DataFrame(self.buffer)
        df.to_csv('{}/out.csv'.format(self.file_path))  
        return df
    
    def load_json(self):
        with open("{}/out.json".format(self.file_path),"r") as jf:
            data = json.load(jf)
        self.buffer = data


