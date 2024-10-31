from .imports import import_torch, import_torchvision
import os

torch = import_torch()


@torch.no_grad()
def gridify(
    tensor: torch.Tensor,
    max_width: int | None = None,
    columns: int | None = None,
    padding: int = 3,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """
    Makes a grid of images/videos from a batch of images.
    Like torchvision's make_grid, but more flexible.
    Accepts (B,\\*,H,W)

    Args:
        tensor : (B,\\*,H,W) tensor
        max_width : max width of the output grid. Resizes images to fit the width
        columns : number of columns of the grid. If None, uses 8 or less
        padding : padding to add to the images
        pad_value : color of the padding

    Returns:
        (\\*,H',W') tensor, representing the grid of images/videos
    """
    transf = import_torchvision().transforms

    B, H, W = tensor.shape[0], tensor.shape[-2], tensor.shape[-1]
    device = tensor.device
    if columns is not None:
        numCol = columns
    else:
        numCol = min(8, B)

    black_cols = (-B) % numCol
    tensor = torch.cat([tensor, torch.zeros(black_cols, *tensor.shape[1:], device=device)], dim=0)  # (B',*,H,W)
    tensor = transf.Pad(padding, fill=pad_value)(tensor)  # (B',*,H+padding*2,W+padding*2)

    B, H, W = tensor.shape[0], tensor.shape[-2], tensor.shape[-1]
    rest_dim = tensor.shape[1:-2]

    rest_dim_prod = 1
    for dim in rest_dim:
        rest_dim_prod *= dim

    if max_width is not None:
        resize_ratio = max_width / (W * numCol)
        if resize_ratio < 1:
            indiv_tens_size = int(H * resize_ratio), int(W * resize_ratio)
            tensor = tensor.reshape((B, rest_dim_prod, H, W))
            tensor = transf.Resize(indiv_tens_size, antialias=True)(tensor)  # (B',rest_dim_prod,H',W')

    B, H, W = tensor.shape[0], tensor.shape[-2], tensor.shape[-1]
    assert B % numCol == 0

    numRows = B // numCol

    tensor = tensor.reshape((numRows, numCol, rest_dim_prod, H, W))  # (numRows,numCol,rest_dim_prod,H',W')
    tensor = torch.einsum("nmrhw->rnhmw", tensor)  # (rest_prod,numRows,H',numCol,W')
    tensor = tensor.reshape((rest_dim_prod, numRows * H, numCol * W))  # (rest_prod,numRows*H,numCol*W)
    tensor = tensor.reshape((*rest_dim, numRows * H, numCol * W))  # (*,numRows*H,numCol*W)

    return tensor


@torch.no_grad()
def _create_folder(folder: str, create_folder: bool = True):
    if create_folder:
        os.makedirs(folder, exist_ok=True)
    else:
        if not (os.path.exists(folder)):
            raise FileNotFoundError(f"Folder {folder} does not exist !")
