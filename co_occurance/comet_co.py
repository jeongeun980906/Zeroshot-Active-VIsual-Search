import spacy
from co_occurance.generate import Comet
from gensim.models import KeyedVectors

class co_occurance_score():
    def __init__(self,device):
        self.nlp = spacy.load('en_core_web_md')
        print("model loading ...")
        DIR = "./co_occurance/comet-atomic_2020_BART"
        self.comet = Comet(DIR,device=device)
        self.comet.model.zero_grad()
        print("model loaded")


    def landmark_init(self,landmark_cat):
        self.landmark_cat = landmark_cat

    def score(self,query_object_name):
        new_query_object_name = ''
        if len(query_object_name)>2:
            for i, letter in enumerate(query_object_name):
                if i and letter.isupper():
                    new_query_object_name += ' '
                new_query_object_name += letter.lower()
        else:
            new_query_object_name = query_object_name

        head = "A {}".format(new_query_object_name).lower()
        rel = ["AtLocation","LocatedNear"]
        query_1 = "{} {} [GEN]".format(head, rel[0])
        # query_2 = "{} {} [GEN]".format(head, rel[1])
        queries = [query_1]#, query_2]
        results = self.comet.generate(queries, decode_method="beam", num_generate=20)
        print(results)
        res = []
        for l in self.landmark_cat:
            sims = []
            for r in results[0]:
                doc1 = self.nlp(r)
                doc2 = self.nlp(l)
                sims.append(doc1.similarity(doc2))
            # for r in results[1]:
            #     doc1 = self.nlp(r)
            #     doc2 = self.nlp(l)
                # sims.append(doc1.similarity(doc2))
            res.append(round(max(sims),3))
        return res


class co_occurance_score_base():
    def __init__(self):
        '''
        Word2Vec
        '''
        # Load vectors directly from the file
        self.model = KeyedVectors.load_word2vec_format('data/GoogleGoogleNews-vectors-negative300.bin', binary=True)
    def landmark_init(self,landmark_cat):
        self.landmark_cat = landmark_cat
    
    def score(self,query_object_name):
        new_query_object_name = ''

        for i, letter in enumerate(query_object_name):
            if i and letter.isupper():
                new_query_object_name += ' '
            new_query_object_name += letter.lower()

        res = []
        for l in self.landmark_cat:
            sim = self.model.similarity(l,new_query_object_name)
            res.append(round(max(sim),3))
        return res