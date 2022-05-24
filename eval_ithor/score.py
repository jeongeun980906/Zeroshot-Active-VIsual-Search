import pandas as pd
import json
import os
class score_storage():
    def __init__(self,args):
        self.buffer = []
        self.file_path = './res/%d_%d_%d'%(args.base_detector,args.co_base,args.num_loi)
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


