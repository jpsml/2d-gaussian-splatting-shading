import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import get_encoder
from .activation import trunc_exp
from .net_init import init_seq
import numpy as np
import os

METALLIC_THRESHOLD = 0.5

class Density(nn.Module):
    def __init__(self, beta=None):
        super().__init__()
        if beta is not None:
            self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, sdf, beta=None):
        return self.density_func(sdf, beta=beta)

class LaplaceDensity(Density):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    def __init__(self, beta, beta_min=0.0001, beta_max=1.0):
        super().__init__(beta)
        self.beta_min = beta_min
        self.beta_max = beta_max

    def density_func(self, sdf, beta=None, alpha=None):
        if beta is None:
            beta = self.get_beta()

        alpha = 1 / beta if alpha is None else alpha
        return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))

    def get_beta(self):
        beta_clamp = torch.clamp(self.beta.detach(), self.beta_min, self.beta_max)
        beta_diff = beta_clamp - self.beta.detach()
        beta = self.beta + beta_diff
        # beta = torch.clamp(self.beta, self.beta_min, self.beta_max)
        return beta #* 0 + 0.02

class NeuralRendererModel(nn.Module):
    def __init__(self,
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 bound=1,
                 num_levels=16,
                 roughness_bias = -1
                 ):
        super().__init__()
        
        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.bound = bound
        # if self.opt.eikonal_loss_weight > 0:
        #     encoding = "hashgrid_diff"
        encoding = 'hashgrid_diff'
        self.encoder, self.in_dim = get_encoder(
                    encoding, level_dim=2,
                    desired_resolution=bound * 2048, 
                    base_resolution=16,
                    num_levels=num_levels, log2_hashmap_size=19,
                    multires=0 # for PE
                )
        self.roughness_bias = roughness_bias

        self.use_sdf = True
        
        beta_param_init = 0.1
        beta_min, beta_max = 0.0005, 1
        self.sdf_density = LaplaceDensity(beta_param_init, beta_min, beta_max)
        
        self.embed_dim = 0
        self.embed_fn = None
        self.geometric_init = False
        if self.geometric_init:
            inside_outside = False # True for indoor scenes
            weight_norm = True
            bias = 1.0
        #     multires = 6
        #     from freqencoder import FreqEncoder
        #     self.embed_fn = FreqEncoder(input_dim=3, degree=multires)
        #     self.embed_dim = self.embed_fn.output_dim
        # else:
        #     self.embed_fn = None
        
        self.w_material = False
        self.in_roughness, self.in_metallic, self.in_base_color = 0, 0, 0
        
        material_dims = self.in_roughness + self.in_metallic + self.in_base_color
        self.embed_dim += material_dims
        self.w_material = material_dims > 0

        sdf_net = []
        self.skip_layers = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim + self.embed_dim
            else:
                in_dim = hidden_dim
            
            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim
                out_dim = out_dim + 2
            elif l+1 in self.skip_layers:
                out_dim = hidden_dim - self.in_dim
            else:
                out_dim = hidden_dim
            lin = nn.Linear(in_dim, out_dim, bias=True)

            if self.geometric_init:
                if l == self.num_layers - 1:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(in_dim), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(in_dim), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)

                elif self.in_dim > 3 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif l in self.skip_layers:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(self.in_dim - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            
                if weight_norm:
                    lin = nn.utils.weight_norm(lin)

            sdf_net.append(lin) # why no bias here?

        self.sdf_net = nn.ModuleList(sdf_net)
        self.sdf_act = nn.ReLU(inplace=True) if not self.geometric_init else nn.Softplus(beta=100)
        
        self.net_act = nn.ReLU(inplace=True)
        self.color_act = nn.Sigmoid()            
            
        init_seq(self.sdf_net, 'xavier_uniform', self.sdf_act)
            
        self.roughness_act = nn.Softplus()

    def forward_geometry(self, xyz, material=None):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]
        # material: dict
        # print("calling forward", file=sys.stderr)

        # sigma
        x = self.encoder(xyz, bound=self.bound)
        
        if self.embed_fn is not None:
            x = torch.cat([self.embed_fn(xyz), x], dim=-1)

        h = x
        for l in range(self.num_layers):
            if l in self.skip_layers:
                h = torch.cat([h, x], dim=-1) / np.sqrt(2)
            h = self.sdf_net[l](h)
            if l != self.num_layers - 1:
                h = self.sdf_act(h)

        #sigma = F.relu(h[..., 0])
        if self.use_sdf:
            sdf = h[..., 0]
            sigma = None
        else:
            sdf = None
            sigma = trunc_exp(h[..., 0])
       
        geo_feat = h[..., 1:1+self.geo_feat_dim]
        geo_feat = F.normalize(geo_feat, dim=-1)
        
        raw_roughness = h[..., 1+self.geo_feat_dim:2+self.geo_feat_dim]
        self.blend_weight = torch.sigmoid(h[..., 2+self.geo_feat_dim:3+self.geo_feat_dim])
        self.roughness = 0.2 * self.roughness_act(raw_roughness + self.roughness_bias)
        self.roughness = self.roughness

        self.metallic = 1.

        return sdf, sigma, geo_feat

    def forward_sigma(self, xyzs, material=None, **kwargs):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]
        # print("calling forward", file=sys.stderr)
        use_sdf_sigma_grad = kwargs.get("use_sdf_sigma_grad", False)

        sdfs, sigmas, geo_feats = self.forward_geometry(xyzs, material)
        normals = None
        eikonal_sdf_gradients = None
        if self.use_sdf:
            if use_sdf_sigma_grad:
                # compute normal
                normals, eikonal_sdf_gradients = self.compute_normal(sdfs, xyzs, True)
            # compute sdfs -> sigmas
            assert sigmas is None
            # dirs, dists, gradients, cos_anneal_ratio=1.0
            sigmas = self.sdf_density(sdfs)
        elif (not self.use_sdf) and use_sdf_sigma_grad:
            normals, eikonal_sdf_gradients = self.compute_normal(sigmas, xyzs, True)
        
        return sdfs, sigmas, geo_feats, normals, eikonal_sdf_gradients

    def density(self, x, **kwargs):
        # x: [N, 3], in [-bound, bound]
        # print("calling density", file=sys.stderr)

        sdf, sigma, geo_feat, normal, eikonal_sdf_gradient = self.forward_sigma(x, **kwargs)

        return {
            'sdf': sdf,
            'sigma': sigma,
            'geo_feat': geo_feat,
            'normal': normal,
            'sdf_gradients': eikonal_sdf_gradient
        }

    # optimizer utils
    def get_params(self, lr, plr=0, slr=0, elr=0):
        plr = lr if plr == 0 else plr
        slr = lr if slr == 0 else slr
        elr = lr if elr == 0 else elr
        params = []
        params = [
            {'params': self.encoder.parameters(), 'lr': plr, "name": "encoder"},
            {'params': self.sdf_net.parameters(), 'lr': lr, "name": "sdf_net"},
            # {'params': self.encoder_dir.parameters(), 'lr': lr},
        ]
        if self.use_sdf:
            params.append(
                {'params': self.sdf_density.parameters(), 'lr': slr, "name": "sdf_density"},
            )

        return params

    def compute_normal(self, sdf_or_sigma, xyzs, get_eikonal_sdf_gradient=False):
        eikonal_sdf_gradient = None
        d_outpus = torch.ones_like(sdf_or_sigma)
        if self.use_sdf:
            normals = torch.autograd.grad(outputs=sdf_or_sigma, inputs=xyzs, grad_outputs=d_outpus, retain_graph=True, create_graph=True)[0]
        else:
            normals = - torch.autograd.grad(outputs=sdf_or_sigma, inputs=xyzs, grad_outputs=d_outpus, retain_graph=True, create_graph=True)[0]
        if get_eikonal_sdf_gradient:
            eikonal_sdf_gradient = normals # for Eikonal loss        
        normals = F.normalize(normals, dim=-1, eps=1e-10)        
        return normals, eikonal_sdf_gradient
    