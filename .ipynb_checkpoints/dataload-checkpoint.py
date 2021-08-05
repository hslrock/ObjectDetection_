import torch
from torch.utils.data import Dataset
import torch.nn as nn
##File Management

import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


#Image,Numpy
import numpy as np
import cv2


#Img Augment
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from torchvision import transforms as T

from utills import ssearch,misc
from utills.misc import create_label,balance_df
import random
def xml_to_csv(pths,img_path):
    '''pths: list of xml_files'''
    CLASS_NAME=['cat','dog']  #We only have two classes, but could be changed in future

    xml_list = []
    for xml_file in pths:
        # Read in the xml file
        tree = ET.parse(xml_file)
        path = os.path.join(img_path, tree.findtext("./filename"))
        height = int(tree.findtext("./size/height"))
        width = int(tree.findtext("./size/width"))
        xmin = int(tree.findtext("./object/bndbox/xmin"))
        ymin = int(tree.findtext("./object/bndbox/ymin"))
        xmax = int(tree.findtext("./object/bndbox/xmax"))
        ymax = int(tree.findtext("./object/bndbox/ymax"))
        name = CLASS_NAME.index(tree.findtext("./object/name"))
        xml_list.append([path,name,xmin,ymin,xmax,ymax])
    col_n = ["filename", "target","xmin", "ymin", "xmax", "ymax"]
    df = pd.DataFrame(xml_list, columns=col_n)
    return df
# class PetData(Dataset):
#     def __init__(self, dataframe,train=False,ssearch=False,samples=16):
#         self.df=dataframe
#         self.ssearch=ssearch
#         self.transform=iaa.Sequential([iaa.Resize((224,224))])
#         self.torch_transform=T.Compose([T.ToTensor(),
#                                         T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])    
#         self.samples=samples
#         self.train=train
#     def __len__(self):
#         return len(self.df)
    
#     def __getitem__(self, idx):
#         regions=None
#         fn,target,xmin,ymin,xmax,ymax=self.df.iloc[idx] #
#         im=cv2.cvtColor(cv2.imread(fn),cv2.COLOR_BGR2RGB) ##Load Img
        
#         class_label=target+1  ##Class #0 represents background
#         bbs=BoundingBoxesOnImage([BoundingBox(xmin,ymin,xmax,ymax,label=class_label)], shape=im.shape) #BBox
#         image_aug, bbs_aug = self.transform(image=im, bounding_boxes=bbs) #Transformation
#         bbs_aug=torch.stack([torch.tensor([bb.x1,bb.y1,bb.x2,bb.y2,bb.label]) for bb in bbs_aug])
        
#         region_np=[]
#         img_shape=image_aug.shape   
        
#         if self.ssearch:                                
#             regions=ssearch.selective_search(image_aug, scale=50, sigma=0.8, min_size=20)
#             if self.train:
#                 regions=create_label(regions,bbs_aug,iou_threshold=0.5)
#                 regions=[dict(t) for t in {tuple(d.items()) for d in regions}]
#                 for dicts in regions:
#                     region_np.append((np.array(dicts['rect'],dtype=np.float)))
#                 region_np=np.stack(region_np)
#                 region_np=region_np[np.where(region_np[:,-1]>0.1)]
                
#                 try:
#                     pos_idx = random.choices(np.where((region_np[:,4]) != 0)[0],k=16)
#                 except:
#                     pos_idx=[]
#                 neg_samples=64
#                 if len(pos_idx):
#                     neg_samples=48
#                 neg_idx = random.choices(np.where((region_np[:,4]) == 0)[0],k=neg_samples)
#                 region_np=region_np[pos_idx+neg_idx]
#                 region_np=torch.from_numpy(region_np)
#                 labels=region_np[:,4].long()
#                 bbox_idx=region_np[:,5].long()
#                 region_np=torch.stack([torch.clamp(region_np[:,0]-16,0,img_shape[1]),
#                                        torch.clamp(region_np[:,1]-16,0,img_shape[0]),
#                                        torch.clamp(region_np[:,2]+16,0,img_shape[1]),
#                                        torch.clamp(region_np[:,3]+16,0,img_shape[0])],dim=1)  
#             else:
#                 regions=[dict(t) for t in {tuple(d.items()) for d in regions}]
#                 for dicts in regions:
#                     region_np.append((np.array(dicts['rect'],dtype=np.float)))
#                 region_np=np.stack(region_np)
                        
#                 region_np=torch.from_numpy(region_np)
#                 region_np=torch.stack([torch.clamp(region_np[:,0]-16,0,img_shape[1]),
#                                        torch.clamp(region_np[:,1]-16,0,img_shape[0]),
#                                        torch.clamp(region_np[:,2]+16,0,img_shape[1]),
#                                        torch.clamp(region_np[:,3]+16,0,img_shape[0])],dim=1)
#                 return self.torch_transform(image_aug), bbs_aug,region_np,

#         return self.torch_transform(image_aug), bbs_aug,region_np,labels,bbox_idx
    

class PetData_FASTRCNN(Dataset):
    def __init__(self, dataframe,train=False,label_included=True,samples=16):
        self.df=dataframe
        self.transform=iaa.Sequential([iaa.Resize((224,224))])
        self.torch_transform=T.Compose([T.ToTensor(),
                                        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])    
        self.samples=samples
        self.label_included=label_included
        self.train=train
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        regions=None
        fn,target,xmin,ymin,xmax,ymax=self.df.iloc[idx] #
        im=cv2.cvtColor(cv2.imread(fn),cv2.COLOR_BGR2RGB) ##Load Img
        
        class_label=target+1  ##Class #0 represents background
        bbs=BoundingBoxesOnImage([BoundingBox(xmin,ymin,xmax,ymax,label=class_label)], shape=im.shape) #BBox
        image_aug, bbs_aug = self.transform(image=im, bounding_boxes=bbs) #Transformation
        bbs_aug=torch.stack([torch.tensor([bb.x1,bb.y1,bb.x2,bb.y2,bb.label]) for bb in bbs_aug])
        regions=ssearch.selective_search(image_aug, scale=50, sigma=0.8, min_size=20)
        region_np=[]
        img_shape=image_aug.shape      

        if self.label_included:
            regions=create_label(regions,bbs_aug,iou_threshold=0.5)
            regions=[dict(t) for t in {tuple(d.items()) for d in regions}]
            for dicts in regions:
                region_np.append((np.array(dicts['rect'],dtype=np.float)))
            region_np=np.stack(region_np)
            if self.train:
                region_np=region_np[np.where(region_np[:,-1]>0.1)]

                try:
                    pos_idx = random.choices(np.where((region_np[:,4]) != 0)[0],k=16)
                except:
                    pos_idx=[]
                neg_samples=64
                if len(pos_idx):
                    neg_samples=48
                neg_idx = random.choices(np.where((region_np[:,4]) == 0)[0],k=neg_samples)
                region_np=region_np[pos_idx+neg_idx]
                #region_np=region_np[torch.randperm(region_np.size()[0])]


            region_np=torch.from_numpy(region_np)
            
            labels=region_np[:,4].long()
            bbox_idx=region_np[:,5].long()
            region_np=torch.stack([torch.clamp(region_np[:,0]-16,0,img_shape[1]),
                                   torch.clamp(region_np[:,1]-16,0,img_shape[0]),
                                   torch.clamp(region_np[:,2]+16,0,img_shape[1]),
                                   torch.clamp(region_np[:,3]+16,0,img_shape[0])],dim=1)
            return self.torch_transform(image_aug), bbs_aug,region_np,labels,bbox_idx
        else:
            regions=[dict(t) for t in {tuple(d.items()) for d in regions}]
            for dicts in regions:
                region_np.append((np.array(dicts['rect'],dtype=np.float)))
            region_np=np.stack(region_np)
            return self.torch_transform(image_aug), bbs_aug,region_np
    
class PetData_FASTERRCNN(Dataset):
    def __init__(self, dataframe,train=False,ssearch=False,samples=16):
        self.df=dataframe
        self.ssearch=ssearch
        self.transform=iaa.Sequential([iaa.Resize({"shorter-side": 600, "longer-side": "keep-aspect-ratio"})])
        self.torch_transform=T.Compose([T.ToTensor(),
                                        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])    
        self.samples=samples
        self.train=train
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        regions=None
        fn,target,xmin,ymin,xmax,ymax=self.df.iloc[idx] #
        im=cv2.cvtColor(cv2.imread(fn),cv2.COLOR_BGR2RGB) ##Load Img
        
        class_label=target+1  ##Class #0 represents background
        bbs=BoundingBoxesOnImage([BoundingBox(xmin,ymin,xmax,ymax,label=class_label)], shape=im.shape) #BBox
        image_aug, bbs_aug = self.transform(image=im, bounding_boxes=bbs) #Transformation
        bbs_aug=torch.stack([torch.tensor([bb.x1,bb.y1,bb.x2,bb.y2,bb.label]) for bb in bbs_aug])
        
        region_np=[]
        img_shape=image_aug.shape      
        return self.torch_transform(image_aug), bbs_aug,
    
class PetData(Dataset):

    
    def __init__(self, dataframe,train=False,tensor_return=True,raw_return=False):
        self.df=dataframe
        self.tensor_return=tensor_return
        self.raw_return=raw_return
        self.transform=iaa.Sequential([iaa.Resize((512,512))])
        self.tensorify = T.ToTensor()
        self.normalize=T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    def __len__(self):
        return len(self.df)

    
    def __getitem__(self, idx):
        fn,target,xmin,ymin,xmax,ymax=self.df.iloc[idx] #
        im=cv2.cvtColor(cv2.imread(fn),cv2.COLOR_BGR2RGB) ##Load Img
        
        
        class_label=target  ##Class
        bbs=BoundingBoxesOnImage([BoundingBox(xmin,ymin,xmax,ymax,label=class_label)], shape=im.shape) #BBox
        image_aug, bbs_aug = self.transform(image=im, bounding_boxes=bbs) #Transformation

        if self.raw_return:
            return self.tensorify(image_aug),torch.stack([torch.tensor([bb.x1,bb.y1,bb.x2,bb.y2,bb.label]) for bb in bbs_aug])
        else:
                return self.normalize(self.tensorify(image_aug)),torch.stack([torch.tensor([bb.x1,bb.y1,bb.x2,bb.y2,bb.label]) for bb in bbs_aug])

#             if self.tensor_return:
#                 return self.torch_transform(im),torch.stack([torch.tensor([bb.x1,bb.y1,bb.x2,bb.y2,bb.label]) for bb in bbs])
        
#         image_aug, bbs_aug = self.transform(image=im, bounding_boxes=bbs) #Transformation
#         image_aug=self.torch_transform(image_aug)
#         if self.tensor_return:
#             bbs_aug=torch.stack([torch.tensor([bb.x1,bb.y1,bb.x2,bb.y2,bb.label]) for bb in bbs_aug])
                
#         return image_aug,bbs_aug,im#,class_label
    
class Sub_region(Dataset):
    def __init__(self, df,img,pil=True,return_idx=False):
        self.src_img=img
        self.df=df
        self.resize=nn.Upsample((224,224))
        self.shape=self.src_img.shape
        self.pil=pil

        self.toPIL=T.ToPILImage()
        self.transforms=T.Compose([
        #T.ToTensor(),
        T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        self.return_idx=return_idx
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        label=self.df.iloc[idx]['labels']
        rect=self.df.iloc[idx]['rect']
        x1=np.clip(rect[0]-16,0,self.shape[2])
        y1=np.clip(rect[1]-16,0,self.shape[1])
        x2=np.clip(rect[2]+16,0,self.shape[2])
        y2=np.clip(rect[3]+16,0,self.shape[1])
        
        img=self.resize(self.src_img[:,y1:y2,x1:x2].unsqueeze(0)).squeeze(0)
      #  img=self.toPIL(img*0.5+0.5)
        
        
        if self.pil:
            return self.transforms(img*0.5+0.5),label
        elif self.return_idx:
            return self.transforms(img),label,idx
        else:
            return self.transforms(img),label    
    
    
class Sub_region_train(Dataset):
    def __init__(self, df,base_path,train=True):
        self.df=df
        self.base_path=base_path
        if train:
            self.transforms=T.Compose([
                 T.RandomHorizontalFlip(),
                T.RandomRotation(10),
                T.ToTensor(),

                T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),

            ])
        else:
            self.transforms=T.Compose([
            T.ToTensor(),
            T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),

            ])
        self.le= preprocessing.LabelEncoder()
        self.le=self.le.fit([-1,0,1])
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
       # print(self.df.filename[idx])
       # img=cv2.imread(os.path.join(self.base_path,self.df.filename[idx]))
     #   img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img=Image.open(os.path.join(self.base_path,self.df.filename[idx]))
        img=self.transforms(img)
        
        return img, self.le.transform([self.df.labels[idx]])[0]
    
    