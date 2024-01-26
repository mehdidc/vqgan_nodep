import torch
from vqgan import VQModel
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
