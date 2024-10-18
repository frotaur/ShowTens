def import_torch():
    """
        Lazily import torch, giving helpful error message if not installed.
    """
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError(
            "PyTorch is not installed. ShowTens requires PyTorch. "
            "Please install it with one of the following commands:\n\n"
            "- For CPU only: pip install torch\n"
            "- For CUDA support: Visit https://pytorch.org for installation instructions "
            "specific to your system and CUDA version."
        )

def import_tranforms():
    """
        Lazily import torchvision.transforms, giving helpful error message if not installed.
    """
    try:
        import torchvision.transforms as transf
        return transf
    except ImportError:
        raise ImportError(
            "torchvision is not installed. ShowTens requires torchvision. "
            "Please install it with the following command:\n\n"
            "pip install torchvision"
        )

torch = import_torch()


@torch.no_grad()
def gridify(tensor,out_size=800,columns=None):
    """
        Makes a grid of images/videos from a batch of images. 
        Like torchvision's make_grid, but more flexible. 
        Accepts (B,*,H,W)

        TODO : add choice of padding size and color.
        
        Args:
            tensor : (B,*,H,W) tensor
            out_size : height of the output grid
            columns : number of columns of the grid
        
        Returns:
            (*,H',W') tensor, representing the grid of images/videos
    """
    transf = import_tranforms()

    B,H,W = tensor.shape[0],tensor.shape[-2],tensor.shape[-1]
    device = tensor.device
    if(columns is not None):
        numCol=columns
    else :
        numCol=min(8,B)


    black_cols = (-B)%numCol
    tensor = torch.cat([tensor,torch.zeros(black_cols,*tensor.shape[1:],device=device)],dim=0) # (B',*,H,W)
    tensor = transf.Pad(3)(tensor) # (B',*,H+3*2,W+3*2)

    B,H,W = tensor.shape[0],tensor.shape[-2],tensor.shape[-1]
    rest_dim = tensor.shape[1:-2]

    rest_dim_prod = 1
    for dim in rest_dim:
        rest_dim_prod*=dim
    
    resize_ratio = out_size/(H*numCol)

    indiv_vid_size = int(H*resize_ratio),int(W*resize_ratio)
    tensor = tensor.reshape((B,rest_dim_prod,H,W))
    tensor = transf.Resize(indiv_vid_size,antialias=True)(tensor) # (B',rest_dim_prod,H',W')
    B,H,W = tensor.shape[0],tensor.shape[-2],tensor.shape[-1]

    assert B%numCol==0

    numRows = B//numCol

    tensor = tensor.reshape((numRows,numCol,rest_dim_prod,H,W)) # (numRows,numCol,rest_dim_prod,H',W')
    tensor = torch.einsum('nmrhw->rnhmw',tensor) # (rest_prod,numRows,H',numCol,W')
    tensor = tensor.reshape((rest_dim_prod,numRows*H,numCol*W)) # (rest_prod,numRows*H,numCol*W)
    tensor = tensor.reshape((*rest_dim,numRows*H,numCol*W)) # (*,numRows*H,numCol*W)

    return tensor

