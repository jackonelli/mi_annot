import torch

vits16 = torch.hub.load("facebookresearch/dino:main", "dino_vits16")
vits16.cuda()

img_size = 256
random_batch = torch.randn((8, 3, 256, 256)).cuda()
res = vits16(random_batch)
print(vits16)
