import spacy
from co_occurance.generate import Comet

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
        queries = []

        new_query_object_name = ''

        for i, letter in enumerate(query_object_name):
            if i and letter.isupper():
                new_query_object_name += ' '
            new_query_object_name += letter.lower()

        head = "A {}".format(new_query_object_name).lower()
        rel = "AtLocation"
        query = "{} {} [GEN]".format(head, rel)
        queries.append(query)
        results = self.comet.generate(queries, decode_method="beam", num_generate=20)
        print(results)
        res = []
        for l in self.landmark_cat:
            sims = []
            for r in results[0]:
                doc1 = self.nlp(r)
                doc2 = self.nlp(l)

                sims.append(doc1.similarity(doc2))
            res.append(round(max(sims),3))
        return res