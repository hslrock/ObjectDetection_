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



class PetData(Dataset):

    
    def __init__(self, dataframe,train=False,tensor_return=True):
        self.df=dataframe
        self.tensor_return=tensor_return
        if train:
            self.transform=iaa.Sequential([iaa.Resize((224,224))])

        else:
            self.transform=iaa.Sequential([iaa.Resize((224,224))])
        
        self.torch_transform = T.Compose([T.ToTensor(),
                                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    def __len__(self):
        return len(self.df)

    
    def __getitem__(self, idx):
        fn,target,xmin,ymin,xmax,ymax=self.df.iloc[idx] #
        im=cv2.cvtColor(cv2.imread(fn),cv2.COLOR_BGR2RGB) ##Load Img
        
        class_label=target  ##Class
        bbs=BoundingBoxesOnImage([BoundingBox(xmin,ymin,xmax,ymax,label=class_label)], shape=im.shape) #BBox

        image_aug, bbs_aug = self.transform(image=im, bounding_boxes=bbs) #Transformation
        image_aug=self.torch_transform(image_aug)
        if self.tensor_return:
            bbs_aug=torch.stack([torch.tensor([bb.x1,bb.y1,bb.x2,bb.y2,bb.label]) for bb in bbs_aug])
                
        return image_aug,bbs_aug#,class_label
    
class Sub_region(Dataset):
    def __init__(self, df,img,pil=True,return_idx=False):
        self.src_img=img
        self.df=df
        self.resize=nn.Upsample((224,224))
        self.transforms=T.ToPILImage()
        self.pil=pil
        self.return_idx=return_idx
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        label=self.df.iloc[idx]['labels']
        rect=self.df.iloc[idx]['rect']
        x1=rect[0]
        y1=rect[1]
        x2=rect[2]+x1+1
        y2=rect[3]+y1+1
        img=self.resize(self.src_img[:,y1:y2,x1:x2].unsqueeze(0))
        if self.pil:
            return self.transforms(img.squeeze(0)*0.5+0.5),label
        elif self.return_idx:
            return img.squeeze(0),label,idx
        else:
            return img.squeeze(0),label


        
    
    
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
    
    