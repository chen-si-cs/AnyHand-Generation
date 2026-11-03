import sys

sys.path.append("./stylegan3/")

import torch
import dnnlib
import numpy as np
import PIL.Image
import legacy

network_pkl = "./manotorch/assets/handy/Handy_Models/hands_texture_model.pkl"
print('Loading networks from "%s"...' % network_pkl)
device = torch.device("cuda")
with dnnlib.util.open_url(network_pkl) as f:
    # print(1)
    G = legacy.load_network_pkl(f)["G_ema"].to(device)  # type: ignore
# print(2)
label = torch.zeros([1, G.c_dim], device=device)


def sample_uv(seed):
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
    img = G(z, label, truncation_psi=1, noise_mode="const")

    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = PIL.Image.fromarray(img[0].cpu().numpy(), "RGB")
    return img


def sample_uv_batch(seeds):
    # seeds: np.array of shape (batch_size,)
    batch_size = seeds.shape[0]

    # Create latent vectors for each seed
    z_list = [np.random.RandomState(int(s)).randn(G.z_dim) for s in seeds]
    z = torch.from_numpy(np.stack(z_list)).to(device)

    # Repeat the label for each sample in the batch
    label_batch = label.repeat(batch_size, 1)

    # Generate images from the latent vectors
    imgs = G(z, label_batch, truncation_psi=1, noise_mode="const")

    # Post-process images to get pixel values
    imgs = (imgs.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

    # Numpy array of shape (batch_size, height, width, 3)
    imgs = imgs.cpu().numpy()

    return imgs
