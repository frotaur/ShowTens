import torch
import os
import matplotlib.pyplot as plt
from torchvision.utils import make_grid



@torch.no_grad()
def showImage(tensor, columns=None, colorbar=False) :
    """"
        Shows tensor as an image using pyplot.
        Any extra dimensions (*,C,H,W) are treated as batch dimensions.

        Args:
        tensor : (H,W) or (C,H,W) or (*,C,H,W) tensor to display
        columns : number of columns to use for the grid of images (default 8 or less)
        colorbar : whether to add a colorbar to the image, only works for grayscale images (default False)
    """
    tensor = tensor.detach().cpu()

    if(len(tensor.shape)==2):
        showImage(tensor[None,:,:],columns,colorbar)
    elif(len(tensor.shape)==3) :
        fig = plt.figure()
        plt.imshow(tensor.permute((1,2,0)))
        plt.axis('off')
        if(tensor.shape[0]==1 and colorbar):
            plt.colorbar()
        # plt.tight_layout(pad=0)
        plt.show()
    elif(len(tensor.shape)==4) :
        # Assume B,C,H,W
        B=tensor.shape[0]
        if(columns is not None):
            numCol=columns
        else :
            numCol=min(8,B)
        
        to_show=make_grid(tensor,nrow=numCol,pad_value=0.2 ,padding=3) # TODO : replace with gridify
        if(tensor.shape[1]==1):
            to_show=to_show.mean(dim=0,keepdim=True)

        showImage(to_show,columns,colorbar)
    elif(len(tensor.shape)>4):
        tensor = tensor.reshape((-1,tensor.shape[-3],tensor.shape[-2],tensor.shape[-1])) # assume all batch dimensions
        print("Assuming extra dimension are all batch dimensions, newshape : ",tensor.shape)
        showImage(tensor,columns,colorbar)
    else :
        raise Exception(f"Tensor shape should be (H,W), (C,H,W) or (*,C,H,W), but got : {tensor.shape} !")

@torch.no_grad()
def saveImage(tensor, folder,name="imagetensor",columns=None,colorbar=False):
    """
        Saves tensor as a png image using pyplot.
        Any extra dimensions (*,C,H,W) are treated as batch dimensions.

        Args:
        tensor : (H,W) or (C,H,W) or (*,C,H,W) tensor to display
        folder : relative path of folder where to save the image
        name : name of the image (do not include extension)
        columns : number of columns to use for the grid of images (default 8 or less)
        colorbar : whether to add a colorbar to the image, only works for grayscale images (default False)
    """
    tensor = tensor.detach().cpu()
    os.makedirs(folder,exist_ok=True)

    if(len(tensor.shape)==2) :
        saveImage(tensor[None,:,:],folder,name,colorbar=colorbar)
    elif(len(tensor.shape)==3) :
        plt.imshow(tensor.permute((1,2,0)))
        plt.axis('off')
        if(tensor.shape[0]==1 and colorbar):
            plt.colorbar()
        plt.savefig(os.path.join(folder,f"{name}.png"),bbox_inches='tight',pad_inches=0)
    elif(len(tensor.shape)==4) :
        # Assume B,C,H,W
        B=tensor.shape[0]
        if(columns is not None):
            numCol=columns
        else :
            numCol=min(8,B)
        to_show=make_grid(tensor,nrow=numCol,pad_value=0.,padding=2) # (3,H',W') # TODO : replace with gridify

        if(tensor.shape[1]==1):# Restore grayscale if it was, make_grid makes it RGB
            to_show=to_show.mean(dim=0,keepdim=True)

        saveImage(to_show,folder,name,colorbar=colorbar)
    elif(len(tensor.shape)>4):
        tensor = tensor.reshape((-1,tensor.shape[-3],tensor.shape[-2],tensor.shape[-1])) # assume all batch dimensions
        print("WARNING : assuming extra dimension are all batch dimensions, newshape : ",tensor.shape)
        saveImage(tensor,folder,name,columns,colorbar=colorbar)
    else :
        raise Exception(f"Tensor shape should be (H,W), (C,H,W) or (*,C,H,W), but got : {tensor.shape} !")
