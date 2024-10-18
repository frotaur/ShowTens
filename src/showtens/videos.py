import os
from .util import gridify, import_torch
import cv2
torch = import_torch()

@torch.no_grad()
def showVideo(tensor, columns=None, fps=30, out_size=800):
    """
        Shows tensor as a video. Accepts both (T,H,W), (T,3,H,W) and (*,T,3,H,W) float tensors.
    
        Args:
        tensor : (T,H,W) or (T,3,H,W) or (*,T,3,H,W) float tensor
        columns : number of columns to use for the grid of videos (default 8 or less)
        fps : fps of the video (default 30)
        out_size : Height of output video (height adapts to not deform videos) (default 800)
    """
    return NotImplementedError("showVideo not implemented yet, use saveVideo instead")

@torch.no_grad()
def saveVideo(tensor,folder,name="videotensor",columns=None,fps=30,out_size=800):
    """
        Saves tensor as a video. Accepts both (T,H,W), (T,3,H,W) and (*,T,3,H,W) float tensors.
        Assumes that the tensor value are in [0,1], clips them otherwise.

        Args:
        tensor : (T,H,W) or (T,3,H,W) or (*,T,3,H,W) float tensor
        folder : path to save the video
        name : name of the video
        columns : number of columns to use for the grid of videos (default 8 or less)
        fps : fps of the video (default 30)
        out_size : Height of output video (height adapts to not deform videos) (default 800)
    """
    tensor = tensor.detach().cpu()
    os.makedirs(folder,exist_ok=True)

    if(len(tensor.shape)<3):
        raise ValueError(f"Tensor shape should be (T,H,W), (T,3,H,W) or (*,T,3,H,W), but got : {tensor.shape} !")
    elif(len(tensor.shape)==3):
        # add channel dimension
        tensor=tensor[:,None,:,:].expand(-1,3,-1,-1) # (T,3,H,W)
        showVideo(tensor,folder,name,columns)
    elif(len(tensor.shape)==4):
        if(tensor.shape[1]==1):
            print('Assuming gray-scale video')
            tensor=tensor.expand(-1,3,-1,-1) # (T,3,H,W)
        assert tensor.shape[1]==3, f"Tensor shape should be (T,H,W), (T,3,H,W) or (*,T,3,H,W), but got : {tensor.shape} !"
        # A single video
        _make_save_video(tensor,folder,name,fps)
    elif(len(tensor.shape)==5):
        if(tensor.shape[2]==1):
            print('Assuming gray-scale video')
            tensor=tensor.expand(-1,-1,3,-1,-1)
        assert tensor.shape[2]==3, f"Tensor shape should be (T,H,W), (T,3,H,W) or (*,T,3,H,W), but got : {tensor.shape} !"
        # Assume B,T,3,H,W
        B,T,C,H,W = tensor.shape

        tensor = gridify(tensor,out_size,columns)

        _make_save_video(video_tens,folder,name,fps)
    elif (len(tensor.shape)>5):
        video_tens = tensor.reshape((-1,*tensor.shape[-4:]))
        showVideo(video_tens,folder,name,columns,fps,out_size)


@torch.no_grad()
def _make_save_video(video_tens,folder,name,fps=30):
    """
        Makes a video in mp4 and saves it at the given folder, with given name.

        Args:
        video_tens : (T,C,H,W) tensor
        path : path to save the video
    """
    T,C,H,W = video_tens.shape
    output_file = os.path.join(folder,f"{name}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_file, fourcc, fps, (W, H))
        
    to_save = (255*video_tens.permute(0,2,3,1).cpu().numpy()).astype(np.uint8)

    for t in range(T):
        frame = to_save[t]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(frame)

    video.release()