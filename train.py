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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
import trimesh
import torch.nn.functional as F
from tqdm import tqdm
from utils.image_utils import psnr, depth2wpos
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from neural_renderer.network import NeuralRendererModel
from neural_renderer.utils import extract_geometry
from torch_ema import ExponentialMovingAverage
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    neural_renderer = NeuralRendererModel().to(device="cuda")
    state_dict = torch.load("ngp_nl_2_gfd_15.pth", map_location="cuda")
    color_state = state_dict['model']
    net_prefix = 'color_net.'
    net_state = {k[len(net_prefix):]: v for k, v in color_state.items() if k.startswith(net_prefix)}
    neural_renderer.color_net.load_state_dict(net_state)
    net_prefix = 'diffuse_net.'
    net_state = {k[len(net_prefix):]: v for k, v in color_state.items() if k.startswith(net_prefix)}
    neural_renderer.diffuse_net.load_state_dict(net_state)
    gaussians = GaussianModel(dataset.sh_degree, neural_renderer)
    scene = Scene(dataset, gaussians, scene_scale=0.8)
    gaussians.training_setup(opt)
    ema_neural_renderer = ExponentialMovingAverage(neural_renderer.parameters(), decay=0.95)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    iter_sdf_start = 120
    #iter_sdf_start = 1800
    #iter_sdf_start = 1

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_sdf_for_log = 0.0
    ema_eikonal_for_log = 0.0
    ema_inter_for_log = 0.0
    ema_neumann_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        neural_renderer.train()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, neural_renderer=neural_renderer)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # regularization
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0

        rend_dist = render_pkg["rend_dist"]
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()

        lambda_sdf = 0.1 if iteration >= iter_sdf_start else 0.0
        lambda_eikonal = 0.001 * (0.01 / 0.001) ** min((iteration - iter_sdf_start) / 60, 1) if iteration >= iter_sdf_start else 0.0
        lambda_inter = 0.01 if iteration >= iter_sdf_start else 0.0
        lambda_neumann = 0.01 if iteration >= iter_sdf_start else 0.0
        #lambda_neumann = 0.0

        unproj_pts = depth2wpos(render_pkg['surf_depth'], viewpoint_cam).permute(1, 2, 0).reshape(-1, 3)
        on_surf_pts = torch.cat([unproj_pts, gaussians.get_xyz])
        on_surf_sdfs, _, _, sdf_normals, eikonal_sdf_gradients = neural_renderer.forward_sigma(on_surf_pts, use_sdf_sigma_grad=True)
        sdf_loss = lambda_sdf * on_surf_sdfs.abs().mean()
        eikonal_loss = lambda_eikonal * ((eikonal_sdf_gradients.norm(p=2, dim=-1) - 1) ** 2).mean()
        off_surf_pts = torch.rand((on_surf_pts.shape[0] // 2, 3), device="cuda") * 2 - 1
        off_surf_sdfs, _, _, _, _ = neural_renderer.forward_sigma(off_surf_pts, use_sdf_sigma_grad=False)
        inter_loss = lambda_inter * torch.exp(-1e2 * torch.abs(off_surf_sdfs)).mean()
        gaussian_normals = torch.cat([rend_normal.permute(1, 2, 0).reshape(-1, 3), gaussians.get_normals])
        #gaussian_normals = rend_normal.permute(1, 2, 0).reshape(-1, 3)
        neumann_loss = lambda_neumann * (1 - F.cosine_similarity(sdf_normals, gaussian_normals, dim=-1)[..., None]).mean()
        #neumann_loss = lambda_neumann * (1 - F.cosine_similarity(sdf_normals[unproj_pts.shape[0], :], gaussian_normals, dim=-1)[..., None]).mean()

        # loss
        total_loss = loss + dist_loss + normal_loss + sdf_loss + eikonal_loss + inter_loss + neumann_loss

        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log
            ema_sdf_for_log = 0.4 * sdf_loss.item() + 0.6 * ema_sdf_for_log
            ema_eikonal_for_log = 0.4 * eikonal_loss.item() + 0.6 * ema_eikonal_for_log
            ema_inter_for_log = 0.4 * inter_loss.item() + 0.6 * ema_inter_for_log
            ema_neumann_for_log = 0.4 * neumann_loss.item() + 0.6 * ema_neumann_for_log


            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "sdf": f"{ema_sdf_for_log:.{5}f}",
                    "eikonal": f"{ema_eikonal_for_log:.{5}f}",
                    "inter": f"{ema_inter_for_log:.{5}f}",
                    "neumann": f"{ema_neumann_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }

                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/sdf_loss', ema_sdf_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/eikonal_loss', ema_eikonal_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/inter_loss', ema_inter_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/neumann_loss', ema_neumann_for_log, iteration)

            #training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))

        training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
        with torch.no_grad():
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                with torch.enable_grad():
                    scene.save(iteration)
                torch.save(neural_renderer.state_dict(), scene.model_path + "/chkpnt_neural_renderer" + str(iteration) + ".pth")
                save_mesh(neural_renderer, os.path.join(scene.model_path, "mesh/iteration_{}/mesh.ply".format(iteration)), threshold=0)

            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                ema_neural_renderer.update()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def save_mesh(model, save_path, resolution=256, threshold=10):
    print(f"==> Saving mesh to {save_path}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def query_func_sdf(pts):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                sdf = model.density(pts.to("cuda"))['sdf']
        return -sdf

    query_func = query_func_sdf

    bound = 1.0
    aabb_infer = torch.FloatTensor([-bound, -bound, -bound, bound, bound, bound])

    vertices, triangles = extract_geometry(aabb_infer[:3], aabb_infer[3:], resolution=resolution, threshold=threshold, query_func=query_func)

    mesh = trimesh.Trimesh(vertices, triangles, process=False) # important, process=True leads to seg fault...
    mesh.export(save_path)

    print(f"==> Finished saving mesh.")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

#@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    if scene.gaussians.neural_renderer is not None:
                        scene.gaussians.neural_renderer.eval()
                    render_pkg = renderFunc(viewpoint, scene.gaussians, neural_renderer=scene.gaussians.neural_renderer, *renderArgs)
                    with torch.no_grad():
                        image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                        if tb_writer and (idx < 5):
                            from utils.general_utils import colormap
                            depth = render_pkg["surf_depth"]
                            norm = depth.max()
                            depth = depth / norm
                            depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                            tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                            try:
                                rend_alpha = render_pkg['rend_alpha']
                                rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                                surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                                tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                                tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                                tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                                rend_dist = render_pkg["rend_dist"]
                                rend_dist = colormap(rend_dist.cpu().numpy()[0])
                                tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                            except:
                                pass

                            if iteration == testing_iterations[0]:
                                tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                        l1_test += l1_loss(image, gt_image).mean().double()
                        psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    # All done
    print("\nTraining complete.")