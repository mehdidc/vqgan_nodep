# VQGAN from Latent Diffusion Models/Taming Transformers without hell of dependencies 

because, we don't need pytorch lightning and all the code base from https://github.com/CompVis/taming-transformers
to load VQGAN.

## Install 

```bash
git clone https://github.com/mehdidc/vqgan_nodep
cd vqgan_nodep
python setup.py develop
```

or simply

`pip install git+https://github.com/mehdidc/vqgan_nodep`

## Usage

to download the model:

```bash
wget https://github.com/mehdidc/feed_forward_vqgan_clip/releases/download/0.1/vqgan_imagenet_f16_16384.yaml
wget https://github.com/mehdidc/feed_forward_vqgan_clip/releases/download/0.1/vqgan_imagenet_f16_16384.ckpt
```

then, to test it:

```python
import torch
from vqgan_nodep import VQModel
from omegaconf import OmegaConf
import torchvision
from PIL import Image
config = OmegaConf.load("vqgan_imagenet_f16_16384.yaml")
model = VQModel(**config.model.params)
model.eval().requires_grad_(False)
model.init_from_ckpt("vqgan_imagenet_f16_16384.ckpt")
img = Image.open("dog.jpg")
img = img.resize((224, 224))
# to Tensor
x = torchvision.transforms.ToTensor()(img)
x = x.unsqueeze(0)
ids = model.tokenize(x)
xr = model.reconstruct_from_tokens(ids)
# to pil
xr = xr.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
xr = (xr * 255).astype("uint8")
xr = Image.fromarray(xr)
xr.save("dog_recon.jpg")
```
