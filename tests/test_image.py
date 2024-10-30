import sys, pathlib, os

sys.path.append(pathlib.Path(__file__).parent.parent.as_posix())

from src.showtens.util import import_torch, import_torchvision

# None implemented yet
from src.showtens import showImage, saveImage

curpath = pathlib.Path(__file__).parent


def test_view():
    """
    Unfortunately I see no way to write automatic unit tests for this function.
    """
    from PIL import Image

    torch = import_torch()
    transf = import_torchvision().transforms
    image = Image.open("whale.png")

    whale = transf.ToTensor()(image)  # (3,H,W);
    randAug = transf.RandomResizedCrop(whale.shape[-2:])
    whale_tile = torch.stack([randAug(whale) for _ in range(9)], dim=0)  # (9,3,H,W)

    alpha_noise = torch.rand(9, 1, *whale.shape[-2:])
    whale_tile = torch.cat([whale_tile, alpha_noise], dim=1)  # (10,4,H,W)
    showImage(whale_tile, columns=5, colorbar=False, max_width=None, padding=0, pad_value=1.0)
    showImage(whale_tile[:, 2:3], columns=None, colorbar=True, max_width=500, padding=3, pad_value=0.0)


def test_save():
    """
    Unfortunately I see no way to write automatic unit tests for this function.
    """
    from PIL import Image

    torch = import_torch()
    transf = import_torchvision().transforms
    image = Image.open(os.path.join(curpath, "test_folder", "whale.png"))

    whale = transf.ToTensor()(image)  # (3,H,W);
    randAug = transf.RandomResizedCrop(whale.shape[-2:])
    whale_tile = torch.stack([randAug(whale) for _ in range(9)], dim=0)  # (10,3,H,W)

    saveImage(
        whale_tile,
        folder="test_folder",
        name="whale_list",
        columns=5,
        colorbar=False,
        max_width=None,
        padding=0,
        pad_value=1.0,
    )
    saveImage(
        whale_tile[:, 2:],
        folder="test_folder",
        name="dawhales",
        columns=None,
        colorbar=True,
        max_width=500,
        padding=3,
        pad_value=0.0,
    )
