import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import get_encoder
from .activation import trunc_exp
from .net_init import init_seq
from .utils import rot_theta
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

def reflect_dir(w_o, normals):
    """Reflect view directions about normals.

    The reflection of a vector v about a unit vector n is a vector u such that
    dot(v, n) = dot(u, n), and dot(u, u) = dot(v, v). The solution to these two
    equations is u = 2 dot(n, v) n - v.

    Caution:
        * This function assumes that the w_o and normals are unit vectors.
        * w_o is direction from the surface to the camera, *unlike* ray_dir in NeRF.

    Args:
        viewdirs: [..., 3] array of view directions.
        normals: [..., 3] array of normal directions (assumed to be unit vectors).

    Returns:
        [..., 3] array of reflection directions, surf to light
    """
    w_r = 2 * torch.sum(w_o * normals, dim=-1, keepdim=True) * normals - w_o
    return w_r

class NeuralRendererModel(nn.Module):
    def __init__(self,
                 encoding_dir="frequency",
                 #num_layers=3,
                 num_layers=2,
                 hidden_dim=64,
                 #geo_feat_dim=12,
                 geo_feat_dim=15,
                 num_layers_color=3,
                 hidden_dim_color=64,
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
        self.use_normal_with_mlp = True
        self.use_reflected_dir = True
        self.use_n_dot_viewdir = True
        
        beta_param_init = 0.1
        beta_min, beta_max = 0.0005, 1
        self.sdf_density = LaplaceDensity(beta_param_init, beta_min, beta_max)

        #self.opacity_weight = nn.Parameter(torch.tensor(1.0))
        
        self.embed_dim = 0
        self.geometric_init = False
        if self.geometric_init:
            inside_outside = False # True for indoor scenes
            weight_norm = True
            bias = 1.0
        
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

        self.diffuse_net = []
        in_dim = self.geo_feat_dim
        in_dim = self.geo_feat_dim + 12

        out_dim = 32
        self.diffuse_net.append(nn.Linear(in_dim, out_dim, bias=True))
        in_dim = out_dim
        diffuse_dim = 3
        self.diffuse_net.append(nn.Linear(in_dim, diffuse_dim, bias=True))
        self.diffuse_net = nn.ModuleList(self.diffuse_net)
        init_seq(self.diffuse_net, 'xavier_uniform', self.net_act)

        # color network
        self.num_layers_color = num_layers_color
        self.hidden_dim_color = hidden_dim_color
        self.in_normal_dim, self.in_refdir_dim = 0, 0

        self.encoder_dir, self.in_dim_dir = get_encoder(encoding_dir, multires=0, degree=5)
        if self.use_normal_with_mlp:
            self.encoder_normal, self.in_normal_dim = get_encoder(encoding_dir, multires=0, degree=5)
        if self.use_reflected_dir:
            self.encoder_refdir, self.in_refdir_dim = get_encoder('integrated_dir', multires=4, degree=5)
            self.diffuse_encoder_refdir, self.in_refdir_dim_diffuse = get_encoder('integrated_dir', multires=4, degree=5)

        # import IPython; IPython.embed()
        self.use_viewdir = False
        if not self.use_viewdir:
            self.in_dim_dir = 0

        self.use_env_net = True
        # TODO: Note that when we learn a obj., we don't have access to env_opt. and env_nets is meaningless.
        # BTW, you can use opt.env_sph_mode to determine whether to use env_nets.
        if self.use_env_net:
            assert self.use_reflected_dir, "use_env_net requires use_reflected_dir"
            self.use_env_net = True
            def get_env_net(in_dim, out_dim, feat_dim, num_layers, bias=True):
                env_net = []
                for l in range(num_layers-1):
                    env_net.append(nn.Linear(in_dim, out_dim, bias=bias))
                    in_dim = out_dim
                env_net.append(nn.Linear(in_dim, feat_dim, bias=bias))
                env_net = nn.ModuleList(env_net)
                init_seq(env_net, 'xavier_uniform', self.net_act)
                return env_net

            self.env_net = get_env_net(self.in_refdir_dim, 256, 12, 4, True)

            self.in_refdir_dim = 12

        color_net =  []

        self.n_dot_viewdir_dim = 1 if self.use_n_dot_viewdir else 0
        # print(f"self.in_normal_dim: {self.in_normal_dim} self.in_dim_dir: {self.in_dim_dir}, self.geo_feat_dim: {self.geo_feat_dim}")
        # print(f"self.in_refdir_dim: {self.in_refdir_dim} self.n_dot_viewdir_dim: {self.n_dot_viewdir_dim}")
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.in_dim_dir + self.geo_feat_dim + self.in_normal_dim + self.in_refdir_dim + self.n_dot_viewdir_dim
            else:
                in_dim = hidden_dim_color

            if l == num_layers_color - 1:
                out_dim = 3
            else:
                out_dim = hidden_dim_color

            color_net.append(nn.Linear(in_dim, out_dim, bias=True))

        self.color_net = nn.ModuleList(color_net)
        init_seq(self.color_net, 'xavier_uniform', self.net_act)
        self.color_net[-1].bias.data -= np.log(3) # make a lower specular at the beginning

    def forward_geometry(self, xyz, material=None):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]
        # material: dict
        # print("calling forward", file=sys.stderr)

        # sigma
        x = self.encoder(xyz, bound=self.bound)

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

    def forward_color(self, geo_feat, d, normal=None, w_r=None, n_dot_w_o=None, n_env_enc=None):
        # color
        h = geo_feat
        env_net = self.env_net
        for l in range(4):
            n_env_enc = env_net[l](n_env_enc)
            if l != 3:
                n_env_enc = self.net_act(n_env_enc)

        n_env_enc = F.normalize(n_env_enc, dim=-1)

        h = torch.cat([h, n_env_enc], dim=-1)

        for l in range(2):
            h = self.diffuse_net[l](h)
            if l != 1:
                h = self.net_act(h)

        self.c_diffuse = self.color_act(h)

        if getattr(self, 'metallic') is not None:
            self.c_diffuse = self.c_diffuse * self.metallic

        # print("not using diffuse only", file=sys.stderr)
        if self.use_viewdir:
            d = self.encoder_dir(d)
            h = torch.cat([d, geo_feat], dim=-1)
        else:
            h = geo_feat

        if self.use_normal_with_mlp:
            assert normal is not None
            h = torch.cat([h, normal], dim=-1)

        branch_dict = {}
        renv_mask, blend_weight = None, 1
        if w_r is not None:
            if self.use_env_net:
                env_net = self.env_net

                for l in range(4):
                    w_r = env_net[l](w_r)
                    if l != 3:
                        w_r = self.net_act(w_r)

                w_r = F.normalize(w_r, dim=-1)

                h_env = torch.cat([h, w_r], dim=-1)
            else:
                h_env = torch.cat([h, w_r], dim=-1)
            branch_dict['env'] = h_env

        if not branch_dict:
            branch_dict['env'] = h

        color_dict = {}
        for k, h_c in branch_dict.items():
            if n_dot_w_o is not None:
                if k == 'renv' and renv_mask is not None:
                    h_c = torch.cat([h_c, n_dot_w_o[renv_mask]], dim=-1)
                else:
                    h_c = torch.cat([h_c, n_dot_w_o], dim=-1)

            for l in range(self.num_layers_color):
                h_c = self.color_net[l](h_c)
                if l != self.num_layers_color - 1:
                    h_c = self.net_act(h_c)

            # sigmoid activation for rgb
            c_specular = self.color_act(h_c)
            color_dict[k] = c_specular


        self.c_specular = color_dict['env']
        if 'renv' in color_dict:
            if renv_mask is not None:
                self.c_specular = self.c_specular.masked_scatter(renv_mask[:, None], self.c_specular[renv_mask] * blend_weight + color_dict['renv'] * (1-blend_weight))
            else:
                self.c_specular = self.c_specular * blend_weight + color_dict['renv'] * (1-blend_weight)

        color = (self.c_diffuse + self.c_specular)

        return color

    # optimizer utils
    def get_params(self, lr, plr=0, slr=0, elr=0):
        plr = lr if plr == 0 else plr
        slr = lr if slr == 0 else slr
        elr = lr if elr == 0 else elr
        params = []
        params = [
            {'params': self.encoder.parameters(), 'lr': plr},
            {'params': self.sdf_net.parameters(), 'lr': lr},
            # {'params': self.encoder_dir.parameters(), 'lr': lr},
        ]
        if self.use_sdf:
            params.append(
                {'params': self.sdf_density.parameters(), 'lr': slr},
            )
        if self.use_env_net:
            if self.env_net is not None:
                params.append({'params': self.env_net.parameters(), 'lr': elr})

        #params.append({'params': self.opacity_weight, 'lr': lr})

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

    def get_color_mlp_extra_params(self, normals, dirs, roughness=0, env_rot_radian=None):
        if normals is None:
            return None, None, None, None
        normals_enc = None
        if self.use_normal_with_mlp:
            normals_enc = self.encoder_normal(normals)

        # Assume dirs is normalized
        w_o = - dirs # unit vector pointing from a point in space to the camera
        w_r_enc = None
        if self.use_reflected_dir:
            w_r = reflect_dir(w_o, normals)
            # TODO: rotate w_r
            if env_rot_radian is not None:
                w_r = w_r @ torch.from_numpy(rot_theta(env_rot_radian)[:3, :3]).float().to(w_r.device)
            w_r_enc = self.encoder_refdir(w_r, roughness=roughness)
        n_dot_w_o = None
        if self.use_n_dot_viewdir:
            n_dot_w_o = torch.sum(normals * w_o, dim=-1, keepdim=True)
        n_env_enc = None

        # TODO: rotate normals
        if env_rot_radian is not None:
            normals = normals @ torch.from_numpy(rot_theta(env_rot_radian)[:3, :3]).float().to(normals.device)
        n_env_enc = self.encoder_refdir(normals, roughness=0.64)

        return normals_enc, w_r_enc, n_dot_w_o, n_env_enc
    