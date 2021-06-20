import numpy as np
import matplotlib.pyplot as plt


def disp_batch(batch,batch_img=False) :
    '''
    batch: One Batch /One Img
    batch_img: if true: select one random input from a batch, else: process one img
    ''' 
    if batch_img:
        image, target, _ = batch
        n = np.random.randint(0, len(image))
        n=0
        im = image[n].permute(1, 2, 0).numpy()
        bbox=np.concatenate((target[n][0][0],target[n][0][1]))
    else:
        image, target, _ = batch
        im=image.permute(1,2,0).numpy()
        bbox=np.concatenate((target[0][0],target[0][1]))
    

    xy=(bbox[0],bbox[1])
    width=bbox[2]-bbox[0]
    height=bbox[3]-bbox[1]
    fig, ax = plt.subplots()
    ax.imshow(np.array(im)*0.5+0.5)
    ax.add_patch(
     patches.Rectangle(
        xy,
        width,
        height,
        edgecolor = 'blue',
        fill=False ) )
    
    plt.show()