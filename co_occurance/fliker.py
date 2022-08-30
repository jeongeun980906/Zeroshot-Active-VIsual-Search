import urllib.request
import numpy as np
import math
import json
'''
[L x O] matrix of co occurance measure
'''
in_landmark_names = ['dining table','sofa','tv monitor']
out_landmark_names = ['desk','drawer','side table','coffee table','bed','arm chair']
landmark_names = in_landmark_names+out_landmark_names

object_names = ['RemoteControl','Laptop','Book','Apple','CD',
            'Pot','Bowl','AlarmClock','TeddyBear',
            'CellPhone','SprayBottle','Pillow']

relations = ['on', 'in', 'contain','support']

class fliker_knowledge():
    def __init__(self,load=True):
        self.co_matrix = np.zeros((len(landmark_names),len(object_names)),dtype=np.float32)
        self.landmark_names = landmark_names
        self.object_names = object_names
        if load:
            self.load()
        else:
            for relation in relations:
                for i,landmark_name in enumerate(landmark_names):
                    new_landmark_name = ''
                    for k, letter in enumerate(landmark_name):
                        if letter == " ":
                            new_landmark_name += '%20'
                        else:
                            new_landmark_name += letter.lower()
                    base_num =self.crawl(new_landmark_name.lower(),relation)
                    for j,query_object_name in enumerate(object_names):
                        new_query_object_name = ''
                        if len(query_object_name)>2:
                            for k, letter in enumerate(query_object_name):
                                if k and letter.isupper():
                                    new_query_object_name += '%20'
                                new_query_object_name += letter.lower()
                        else:
                            new_query_object_name = query_object_name
                        num = self.crawl(new_landmark_name.lower(),new_query_object_name,relation)
                        self.co_matrix[i,j] += num/base_num
                        print(num/base_num)
            self.co_matrix = np.clip(self.co_matrix,0,1)
    
    def landmark_init(self,visible_landmarks):
        self.visible_landmarks = visible_landmarks

    def score(self,query_object_name):
        obj_index = self.object_names.index(query_object_name)
        res = []
        for landmark in self.visible_landmarks:
            lm_index = self.landmark_names.index(landmark)
            res.append(self.co_matrix[lm_index,obj_index])
        # res_num = sum(res)
        # res = [den/res_num for den in res]
        return res
    def load(self):
        with open("./co_occurance/filker.json",'r') as jf:
            data = json.load(jf)
        self.co_matrix = np.asarray(data['data'])

    def save(self):
        res = self.co_matrix.tolist()
        data = dict(data=res)
        with open("./co_occurance/filker.json",'w') as jf:
            json.dump(data,jf,indent=4)

    def web_format(self,landmark_name,query_object=None,relation='on'):
        if query_object != None:
            return "https://www.flickr.com/search/?text={}%20{}%20the%20{}".format(query_object,relation,landmark_name)
        else:
            return "https://www.flickr.com/search/?text={}%20the%20{}".format(relation,landmark_name)

    def crawl(self,landmark_name,query_object=None,relation='on'):
        print(landmark_name,query_object)
        url_name = self.web_format(landmark_name,query_object,relation)
        fp = urllib.request.urlopen(url_name)
        mybytes = fp.read()

        mystr = mybytes.decode("utf8")
        fp.close()
        try:
            a = mystr.split("view-more-link")[1]
            a = a.split("View all")[1]
            a = a.split("<")[0]
            a = a.replace(",","")
            num = int(a)
            print(num)
            return num
        except:
            return 0

if __name__ == '__main__':
    f = fliker_knowledge(load=False)
    f.save()
    print(f.co_matrix)