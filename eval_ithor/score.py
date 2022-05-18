import pandas as pd

class score_storage():
    def __init__(self):
        self.buffer = []

    def append(self,score,query_object_name,scene_type='all'):
        self.buffer.append(dict(score=score,query_name=query_object_name,
                        scene_type=scene_type))
    def average(self):
        df = pd.DataFrame(self.buffer)
        df.to_csv('./res/out.csv')  
        return df


