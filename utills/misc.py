import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
