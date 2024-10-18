# ShowTens : visualize torch tensors EASILY

ShowTens is a simple pytorch package that allows painless and flexible visualizations of image and video tensors.

\<ADD VISUALIZATION VIDEO HERE\>

## Installation

`pip install showtens`

Make sure `torch`and `torchvision` are installed, as the package depends on them.

## Usage
```python
import torch
from showTens import showImage

image1 = torch.rand((3,100,100)) # (C,H,W) image
showImage(image1) # Displays the image using matplotlib
image2 = torch.rand((4,4,3,100,100)) # (B1,B2,C,H,W), two batch dimensions
showImage(image2,colums=4) # Will display as a 4*4 grid

from showTens import saveImage
saveImage(tensor=image1,folder='saved_images',name='imagetensor')

from showTens import saveVideo
video1 = torch.rand((60,3,200,200))
saveVideo(tensor=video1,folder='save_videos',name='videotensor',fps=30)
video2 = torch.rand((4,3,200,200)) # (B,T,C,H,W), batch of videos
saveVideo(tensor=video2,folder='save_videos',name='videobatch',fps=30,columns=2) # 2*2 video grid
```