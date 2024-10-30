import sys,pathlib
sys.path.append(pathlib.Path(__file__).parent.parent.as_posix())

from src.showtens.util import gridify,import_torch,import_torchvision
# None implemented yet
from src.showtens import showImage

def test_grid_pytorch():
    from PIL import Image
    torch = import_torch()
    transf = import_torchvision().transforms

    torchgrid = import_torchvision().utils.make_grid

    image = Image.open("whale.png")
    whale = transf.ToTensor()(image) # (3,H,W)
    randAug = transf.RandomResizedCrop(whale.shape[-2:])

    whale_tile = torch.stack([randAug(whale) for _ in range(9)],dim=0) # (10,3,H,W)

    grid = gridify(whale_tile,columns=5, pad_value=1., padding=0) # (3,H',W')
    gridtorch = torchgrid(whale_tile,nrow=5,padding=0,pad_value=0.) # (3,H',W')

    assert torch.allclose(grid-gridtorch,torch.tensor(0.)), f"My grid does not correspond to pytorch make_grid in the padding 0 case. Max_diff = {torch.max(torch.abs(grid-gridtorch))}"


def test_grid_size():
    torch = import_torch()
    tens_size = [8,32,64]
    padding = [3,5,8]
    num_cols = [3,5,8]
    for sizu, paddingu,colu in zip(tens_size,padding,num_cols):
        tensor = torch.rand(10,3,sizu,sizu)
        grid = gridify(tensor,columns=colu,max_width=sizu//2*colu,padding=0)

        assert grid.shape[-1]==sizu//2*colu, "Grid does not have the right width"

        max_width = 104
        grid = gridify(tensor,columns=colu,max_width=max_width,padding=3)

        new_W = (sizu+paddingu*2)
        reshape_ratio = max_width/(new_W*colu)
        if(reshape_ratio<1):
            image_size = int((new_W)*reshape_ratio)
        else:
            image_size = new_W

        assert grid.shape[-1]==image_size*colu, f"Expected width {image_size*colu}, but got {grid.shape[-1]}"
