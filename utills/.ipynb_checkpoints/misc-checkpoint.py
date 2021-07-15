import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from metrics import iou
import pandas as pd
def disp_batch(batch,batch_img=False,tensor=True) :
    '''
    batch: One Batch /One Img
    batch_img: if true: select one random input from a batch, else: process one img
    ''' 
    target=None
    if batch_img:
        image, target= batch
        n = np.random.randint(0, len(image))
        n=0
        im = image[n].permute(1, 2, 0).numpy()
        bbox=np.concatenate((target[n][0][0],target[n][0][1]))
    else:
        im = batch
        if tensor:
            im=im.permute(1,2,0).numpy()
        if target is not None:
            bbox=np.concatenate((target[0][0],target[0][1]))
    fig, ax = plt.subplots()
    ax.imshow(np.array(im)*0.5+0.5)
    if target is not None:
        xy=(bbox[0],bbox[1])
        width=bbox[2]-bbox[0]
        height=bbox[3]-bbox[1]

        ax.add_patch(
         patches.Rectangle(
            xy,
            width,
            height,
            edgecolor = 'blue',
            fill=False ) )

    plt.show()

    
def create_label(proposed_regions,bboxes):
    ''' 
    Adding Labelling to the proposed regions, if iou of prposed_region and actual bounding box of label 'x' is larger than the threshold, we label it as 'x' 
    
    proposed_regions:output of selective_search
    bboxes: ground truth bbox of original img
    '''
    
    iou_threshold=0.6
    for region in proposed_regions:
        max_iou=0
        region['labels']=-1
        region_tensor=torch.tensor((region['rect'][0],region['rect'][1],region['rect'][0]+region['rect'][2],region['rect'][1]+region['rect'][3]))
        for bbox in bboxes:
            box_tensor=torch.tensor((bbox[0],bbox[1],bbox[2],bbox[3]))
            cur_iou=iou.torch_getIOU(region_tensor,box_tensor)
            if cur_iou>max_iou:
                max_iou=cur_iou
                if max_iou>iou_threshold:
                    region['labels']=bbox[4].item()    
                
        #region['iou']=max_iou
    return proposed_regions

def balance_df(proposed_regions):
    '''
    Rather than we oversample, we tried undersampling the background cases,  
    (*this is only for possible for this dataset which always and only have one of its classes (dog,cat) in the image)
    '''
    regions_df=pd.DataFrame.from_dict(proposed_regions)
 #   g = proposed_regions.groupby('labels')
  #  g = pd.DataFrame(g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True)))
#    
    def sampling_k_elements(group, k=1):
        if len(group) < k:
            return group
        return group.sample(k)
    #return g
    g = regions_df.groupby('labels').apply(sampling_k_elements).reset_index(drop=True)
    
    return g