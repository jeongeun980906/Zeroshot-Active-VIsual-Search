import urllib.request
import numpy as np
import math

'''
[L x O] matrix of co occurance measure
'''
Word_Dict = {
    'Bed': 'bed', 'DiningTable': 'dining table', 'StoveBurner': 'stove', 'Toilet': 'toilet', 'Sink': 'sink',
    'Desk': 'Desk', 'CounterTop':'kitchen table', 'Sofa':'sofa', 'Television':'television','Drawer':'drawer',
     'SideTable':'side table', 'CoffeeTable':'coffee table','ShelvingUnit':'shelving unit','ArmChair':'arm chair','TVStand':'tv stand',
     'Shelf':'shelf'}

relations = ['on', 'in', 'contain','support']

class fliker_knowledge():
    def __init__(self,landmark_names,object_names):
        self.co_matrix = np.zeros((len(landmark_names),len(object_names)),dtype=np.float32)
        self.landmark_names = landmark_names
        self.object_names = object_names
        for relation in relations:
            for i,landmark_name in enumerate(landmark_names):
                landmark_name = Word_Dict[landmark_name]
                landmark_name = landmark_name.replace(" ","%20")
                base_num =self.crawl(landmark_name.lower(),relation)
                for j,query_object_name in enumerate(object_names):
                    new_query_object_name = ''
                    if len(query_object_name)>2:
                        for k, letter in enumerate(query_object_name):
                            if k and letter.isupper():
                                new_query_object_name += '%20'
                            new_query_object_name += letter.lower()
                    else:
                        new_query_object_name = query_object_name
                    num = self.crawl(landmark_name.lower(),new_query_object_name,relation)
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
    f = fliker_knowledge()
    print(f.co_matrix)