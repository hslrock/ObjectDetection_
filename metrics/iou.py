import torch

def torch_getIntersectArea(boxA,boxB):
    #Will return none if no intersect
    dx = min(boxA[2], boxB[2]) - max(boxA[0], boxB[0])
    dy = min(boxA[3], boxB[3]) - max(boxA[1], boxB[1])

    if (dx>=0) and (dy>=0):
        return float(dx*dy)
def torch_getArea(box):
    return float((box[2] - box[0] ) * (box[3] - box[1]))
def torch_getUnion(boxA,boxB,inter_area):
    return torch_getArea(boxA)+torch_getArea(boxB)-inter_area

def torch_getIOU(boxA,boxB):
    I=torch_getIntersectArea(boxA,boxB)

    if I is None: 
        return 0
    U=torch_getUnion(boxA,boxB,I)
   # return float(I)/float(U)

    return torch.div(I,U)