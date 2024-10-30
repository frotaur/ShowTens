import os
import matplotlib.pyplot as plt
from .imports import import_torch, import_torchvision
from .util import gridify, _create_folder

torch = import_torch()
torchvision = import_torchvision()


@torch.no_grad()
def show_image(
    tensor: torch.Tensor,
    columns: int | None = None,
    colorbar: bool = False,
    max_width: int | None = None,
    padding: int = 3,
    pad_value: float = 0.0,
):
    """ "
    Shows tensor as an image using pyplot.
    Any extra dimensions (*,C,H,W) are treated as batch dimensions.

    Args:
    tensor : (H,W) or (C,H,W) or (*,C,H,W) tensor to display
    columns : number of columns to use for the grid of images (default 8 or less)
    colorbar : whether to add a colorbar to the image, only works for grayscale images (default False)
    max_width : maximum width of the image
    padding : number of pixels between images in the grid
    pad_value : inter-padding value for the grid of images
    """
    tensor = _format_image(
        tensor, columns=columns, max_width=max_width, padding=padding, pad_value=pad_value
    )  # (C,H',W') ready to show

    plt.imshow(tensor.permute((1, 2, 0)))
    plt.axis("off")
    if tensor.shape[0] == 1 and colorbar:
        plt.colorbar()
    plt.show()


@torch.no_grad()
def save_image(
    tensor: torch.Tensor,
    folder: str,
    name: str = "imagetensor",
    columns: int | None = None,
    colorbar: bool = False,
    max_width: int | None = None,
    padding: int = 3,
    pad_value: float = 0.0,
    create_folder: bool = True,
):
    """
    Saves tensor as a png image using pyplot.
    Any extra dimensions (*,C,H,W) are treated as batch dimensions.

    Args:
    tensor : (H,W) or (C,H,W) or (*,C,H,W) tensor to display
    folder : relative path of folder where to save the image
    name : name of the image (do not include extension)
    columns : number of columns to use for the grid of images (default 8 or less)
    colorbar : whether to add a colorbar to the image, only works for grayscale images (default False)
    max_width : maximum width of the image
    padding : number of pixels between images in the grid
    pad_value : inter-padding value for the grid of images
    create_folder : whether to create the folder if it does not exist (default True)
    """
    _create_folder(folder, create_folder)

    tensor = _format_image(
        tensor, columns=columns, max_width=max_width, padding=padding, pad_value=pad_value
    )  # (C,H',W') ready to save
    plt.imshow(tensor.permute((1, 2, 0)))
    plt.axis("off")
    if tensor.shape[0] == 1 and colorbar:
        plt.colorbar()

    plt.savefig(os.path.join(folder, f"{name}.png"), bbox_inches="tight", pad_inches=0)


@torch.no_grad()
def _format_image(tensor, columns=None, max_width=None, padding=3, pad_value=0.0):
    """ "
    Shows tensor as an image using pyplot.
    Any extra dimensions (*,C,H,W) are treated as batch dimensions.

    Args:
    tensor : (H,W) or (C,H,W) or (*,C,H,W) tensor to display
    columns : number of columns to use for the grid of images (default 8 or less)
    max_width : maximum width of the image
    padding : number of pixels between images in the grid
    pad_value : inter-padding value for the grid of images
    """
    tensor = tensor.detach().cpu()

    extra_params = dict(columns=columns, max_width=max_width, pad_value=pad_value, padding=padding)
    if len(tensor.shape) == 2:
        # Add batch and channel dimensions
        return _format_image(tensor[None, :, :], **extra_params)
    elif len(tensor.shape) == 3:
        # Reached (C,H,W)
        return tensor
    elif len(tensor.shape) == 4:
        # Gridify assuming (B,C,H,W)
        B = tensor.shape[0]
        if columns is not None:
            numCol = columns
        else:
            numCol = min(8, B)
        tensor = gridify(tensor, columns=numCol, max_width=max_width, pad_value=pad_value, padding=padding)

        return _format_image(tensor, **extra_params)
    elif len(tensor.shape) > 4:
        # Collapse extra dimension to batch
        tensor = tensor.reshape(
            (-1, tensor.shape[-3], tensor.shape[-2], tensor.shape[-1])
        )  # assume all batch dimensions
        print("Assuming extra dimension are all batch dimensions, newshape : ", tensor.shape)
        return _format_image(tensor, **extra_params)
    else:
        raise Exception(f"Tensor shape should be (H,W), (C,H,W) or (*,C,H,W), but got : {tensor.shape} !")
