import clip
from PIL import Image
import numpy as np
import cv2
import torch

class matcher:
    def __init__(self,query_object_name, threshold = 28, device='cuda'):
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
        self.device = device
        self.thres = threshold
        self.tokenize_query(query_object_name)
        # self.clip_model.eval()

    def tokenize_query(self,query_object_name):
        new_query_object_name = ''
        for i, letter in enumerate(query_object_name):
            if i and letter.isupper():
                new_query_object_name += ' '
            new_query_object_name += letter.lower()

        text = clip.tokenize(["a photo of a {}".format(new_query_object_name)]).to(self.device)
        self.text_features = self.clip_model.encode_text(text)
        print("a photo of a {}".format(new_query_object_name))
    
    def make_patch(self,image,bboxs):
        res = []
        vis = []
        bboxs = np.asarray(bboxs,dtype=np.int16)
        for bbox in bboxs:
            y_u = bbox[2]
            y_b = bbox[0]
            x_r = bbox[3]
            x_l = bbox[1]
            crop = image[x_l:x_r,y_b:y_u,:]
            # Make boarder and resize
            y = y_u-y_b
            x = x_r-x_l
            length = max(x, y)

            top = int(length/2 - x/2)
            bottom = int(length/2 - x/2)
            left = int(length/2 - y/2)
            right = int(length/2 - y/2)

            borderType = cv2.BORDER_CONSTANT
            crop = cv2.copyMakeBorder(crop, top, bottom, left, right, borderType)
            crop = cv2.resize(crop,(256,256))
            # convert from BGR to RGB
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            vis.append([crop])
            # convert from openCV2 to PIL
            pil_image=Image.fromarray(crop)
            temp = self.clip_preprocess(pil_image).unsqueeze(0)
            res.append(temp)
        if len(res) == 0:
            return res,vis
        return torch.cat(res,dim=0),np.concatenate(vis,axis=0)
    
    def matching_score(self,img,pred_boxes):
        patches,vis_patches = self.make_patch(img,pred_boxes.tensor.cpu().numpy())
        if len(patches) == 0:
            return [],[]
        image_features = self.clip_model.encode_image(patches.to(self.device))
        dis = torch.matmul(self.text_features,image_features.T)
        print(dis)
        index = torch.where(dis>self.thres)[1]
        show_patch = vis_patches[index.cpu().numpy()]
        candidate_box = pred_boxes[index]
        return show_patch,candidate_box.tensor.cpu().numpy()