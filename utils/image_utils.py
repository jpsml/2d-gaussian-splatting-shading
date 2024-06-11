#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from utils.graphics_utils import fov2focal

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def depth2wpos(depth, camera):
    camD = depth.permute([1, 2, 0])
    shape = camD.shape
    device = camD.device
    h, w, _ = torch.meshgrid(torch.arange(0, shape[0]), torch.arange(0, shape[1]), torch.arange(0, shape[2]), indexing='ij')
    h = h.to(torch.float32).to(device)
    w = w.to(torch.float32).to(device)
    p = torch.cat([w, h], axis=-1)

    p[..., 0:1] -= camera.image_width / 2
    p[..., 1:2] -= camera.image_height / 2
    p *= camD
    K00 = fov2focal(camera.FoVy, camera.image_height)
    K11 = fov2focal(camera.FoVx, camera.image_width)
    K = torch.tensor([K00, 0, 0, K11]).reshape([2,2])
    Kinv = torch.inverse(K).to(device)
    p = p @ Kinv.t()
    camPos = torch.cat([p, camD], -1)

    pose = camera.world_view_transform.to(device)
    Rinv = pose[:3, :3]
    t = pose[3:, :3]
    camWPos = (camPos - t) @ Rinv.t()

    camWPos = (camWPos[..., :3]).permute([2, 0, 1])

    return camWPos
