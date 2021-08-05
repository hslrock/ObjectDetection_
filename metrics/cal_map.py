import pandas as pd
import torch
import numpy as np
from  .iou import  torch_getIOU
def cal_map_(true_data,pred_data,labels,iou_threshold=0.8,debug=False):
    ap = torch.zeros((len(labels)))
    eval_df=None
    for label_idx,label in enumerate(labels):

        #Get detected box/ground_truth for a single label

        pred_images=[p for p in pred_data if p[4]==label]
        gths_images=[g for g in true_data if g[4]==label]

        #Sore detected images by its prevision

        pred_images=(sorted(pred_images, key = lambda x: (x[5]), reverse=True))

        
        #Create flag ,TP,FP for matched true bbox
        num_gt=len(gths_images)
        num_pd=len(pred_images)

        TP=torch.zeros(num_pd)
        FP=torch.ones(num_pd)
        gths_check=[0]*len(gths_images)
        def get_same_image(dets,img_idx):
            gt=[]
            idx_list=[]
            for idx,det in enumerate(dets):
                if det[-1]==img_idx:
                    gt.append(det)
                    idx_list.append(idx)

            return gt,idx_list
        # Matching each detected bbox of a class
        eval_dataframe_class=[{'Class':f'{label_idx}','Image':f'Image {int(data[-1])}', 
                               'Detection':f'P{i}', 'Confidence \%': int(data[-2].item()*100)} for i,data in enumerate(pred_images)] 
        for d_index,det in enumerate(pred_images):
            
            gths,check_idxs = get_same_image(gths_images,det[-1]) #get ground truths from image that belongs to same image as detected box

            
            maxIou=torch.tensor(0)
            #Find Maximum matching iou box
            for gt,check_idx in zip(gths,check_idxs):

                iou=torch_getIOU(det[0:4],gt[0:4])
               # print(gt)
               # print(det)
               # draw_box(np.ones((224,224,3)),
               #          [BoundingBox(det[0:4],label=0)]+[BoundingBox(*ground[0:4],label=1) for ground in gths])
                if iou > maxIou:
                    maxIou = iou
                    erase_idx = check_idx
                           
            if maxIou >=iou_threshold:        # If iou> threshold 
                eval_dataframe_class[d_index]['ioU']='{:.2f}'.format(maxIou.item())
                #f'>{iou_threshold}'
                if gths_check[erase_idx]==0:  # and if unmatched yet
                    TP[d_index]=1             # the bbox is true positive 
                    gths_check[erase_idx]=1   # flag gt_box as matched 
                    eval_dataframe_class[d_index]['Ground Truth']=f'GT{erase_idx}'
                    eval_dataframe_class[d_index]['TP/FP']="TP"
                else:                         # else <threshold if already matched, it is false positive
                    eval_dataframe_class[d_index]['Ground Truth']=f'GT{erase_idx}'
                    eval_dataframe_class[d_index]['TP/FP']="FP"

            else:
                eval_dataframe_class[d_index]['ioU']='{:.2f}'.format(maxIou.item())
                eval_dataframe_class[d_index]['Ground Truth']=f'-'
                eval_dataframe_class[d_index]['TP/FP']="FP"
                
        FP=FP-TP
        acc_FP = torch.cumsum(FP, dim=0)
        acc_TP = torch.cumsum(TP, dim=0)
        rec = acc_TP / (num_gt+ 1e-10)
        prec = (acc_TP/ (acc_FP + acc_TP+1e-10)) #Prevent Division by zero
        
        eval_dataframe_class=pd.DataFrame.from_dict(eval_dataframe_class).join(pd.DataFrame({"Acc TP":acc_TP.tolist(),
                                                            "Acc FP":acc_FP.tolist(),
                                                            "Precision":prec.tolist(),
                                                            "Recall":rec.tolist()}))

        
        ##11-point https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection
        recall_thresholds = torch.arange(start=0, end=1.1, step=.1)
        precisions = torch.zeros((len(recall_thresholds)))
        for i, t in enumerate(recall_thresholds):
            recalls_above_t = rec >= t
            if recalls_above_t.any():
                precisions[i] = prec[recalls_above_t].max()
            else:
                precisions[i] = 0.
      #  plt.figure()        
       # plt.plot(recall_thresholds,precisions)
        ap[label_idx] = precisions.mean()  
        
        if eval_df is None:
            eval_df=eval_dataframe_class
        else:
            eval_df = pd.concat([eval_df,eval_dataframe_class],ignore_index=True)
    return ap,eval_df,recall_thresholds.tolist(),precisions.tolist()