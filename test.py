import torch
from vqgan import VQModel
from omegaconf import OmegaConf
config = OmegaConf.load("vqgan_imagenet_f16_16384.yaml")
model = VQModel(**config.model.params)
model.eval().requires_grad_(False)
model.init_from_ckpt("vqgan_imagenet_f16_16384.ckpt")
x = torch.randn(1, 3, 256, 256)
ids = model.tokenize(x)
xr = model.reconstruct_from_tokens(ids)
print(x.shape, xr.shape)
