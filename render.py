
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

import cv2
import numpy as np
from os import makedirs
import torchvision
import copy
from scipy.constants import gas_constant
import torch.nn.functional as F
from scipy.spatial import cKDTree

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp,dir_flag_name):

    if dir_flag_name == "all" or dir_flag_name == "All":
        render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
        gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
        points_image_path = os.path.join(model_path, name, "ours_{}".format(iteration),"points")
        depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
        point_mask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "point_mask")
        depth1_value_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth1_value")
        depth2_value_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth2_value")
        add_point_path = os.path.join(model_path, name, "ours_{}".format(iteration), "add_point_mask")
    else:
        render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "{}_renders".format(dir_flag_name))
        gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
        points_image_path = os.path.join(model_path, name, "ours_{}".format(iteration), "{}_points".format(dir_flag_name))
        depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "{}_depth".format(dir_flag_name))
        point_mask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "{}_point_mask".format(dir_flag_name))
        depth1_value_path = os.path.join(model_path, name, "ours_{}".format(iteration), "{}_depth1_value".format(dir_flag_name))
        depth2_value_path = os.path.join(model_path, name, "ours_{}".format(iteration), "{}_depth2_value".format(dir_flag_name))
        add_point_path = os.path.join(model_path, name, "ours_{}".format(iteration), "{}_add_point_mask".format(dir_flag_name))

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(points_image_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(point_mask_path, exist_ok=True)
    makedirs(depth1_value_path, exist_ok=True)
    makedirs(depth2_value_path, exist_ok=True)
    makedirs(add_point_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg_my = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, use_render=True)
        rendering= render_pkg_my["render"]
        depthing= render_pkg_my["depth"]
        depth_need_value= render_pkg_my["depth_need_value"]

        screenspace_2D_points = render_pkg_my["2D_point"]
        gt = view.original_image[0:3, :, :]

        mask_image = depthing.clone().zero_() 
        W = mask_image.shape[2]
        H = mask_image.shape[1]
        max_x = torch.max(screenspace_2D_points[:, 0])
        max_y = torch.max(screenspace_2D_points[:, 1])
        min_x = torch.min(screenspace_2D_points[:, 0])
        min_y = torch.min(screenspace_2D_points[:, 1])
        x_coords = screenspace_2D_points[:, 0].clamp(0, mask_image.shape[2] - 1).long() 
        y_coords = screenspace_2D_points[:, 1].clamp(0, mask_image.shape[1] - 1).long()

        mask_image[0, y_coords, x_coords] = 1 

        depth1_image = depthing.clone().zero_()  
        depth1_image[0, y_coords, x_coords] = depth_need_value[:,0]

        depth2_image = depthing.clone().zero_()  
        depth2_image[0, y_coords, x_coords] = depth_need_value[:,1]

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(depthing, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(mask_image, os.path.join(point_mask_path, '{0:05d}'.format(idx) + ".png"))

        torchvision.utils.save_image(depth1_image, os.path.join(depth1_value_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(depth2_image, os.path.join(depth2_value_path, '{0:05d}'.format(idx) + ".png"))

        gt_image_before = view.original_image.cuda()
        gt_image_cpu = gt_image_before.cpu().numpy()
        gt_image_cpu = np.transpose(gt_image_cpu, (1, 2, 0))  
        gt_image_cpu = np.clip(gt_image_cpu * 255, 0, 255)  
        gt_image_cpu = gt_image_cpu.astype(np.uint8)
        screenspace_2D_points_clone = screenspace_2D_points.clone().detach()
        screenspace_2D_points_cpu = screenspace_2D_points_clone.cpu().numpy()
        if gt_image_cpu.ndim == 3 and gt_image_cpu.shape[2] == 3:
            pass  
        else:
            raise ValueError("The image should have 3 channels (RGB).")


        overlay = gt_image_cpu.copy()  
        alpha = 0.7 
        for point in screenspace_2D_points_cpu:

            cv2.circle(overlay, (int(point[0]), int(point[1])), 1, (0, 0, 255), 1)  

        cv2.addWeighted(overlay, alpha, gt_image_cpu, 1 - alpha, 0, gt_image_cpu)

        gt_image_after = torch.tensor(gt_image_cpu).permute(2, 0, 1)  
        gt_image_after = gt_image_after.float() / 255.0  
        torchvision.utils.save_image(gt_image_after, os.path.join(points_image_path, '{0:05d}'.format(idx) + ".png"))

        is_useful = True
        if is_useful == True:
            # TODO_czy
            add_point_pkg = gaussians.add_gaussians_my(render_pkg_my, is_add=False,is_return=True)
            mask = add_point_pkg["add_point_mask"]
            torchvision.utils.save_image(mask, os.path.join(add_point_path, '{0:05d}'.format(idx) + ".png"))


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        point_cloud_path = os.path.join(scene.model_path, "point_cloud/iteration_{}".format(iteration))
        load_path = os.path.join(point_cloud_path, "point_type.txt")

        if os.path.exists(load_path):
            point_type_data = np.loadtxt(load_path, dtype=int) 
            gaussians.point_type = torch.tensor(point_type_data, dtype=torch.int64, device="cuda")
        else:
            num_points = gaussians.get_xyz.shape[0]  
            gaussians.point_type = torch.zeros(num_points, dtype=torch.int64, device="cuda")

        gaussians.point_type_hold = torch.zeros_like(gaussians.point_type)
        print(gaussians.point_type.shape) 
        gaussians.print_infor()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        dir_flag_name = "all"
        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp,dir_flag_name)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp,dir_flag_name)


        is_only_allpoint_gs = False
        if is_only_allpoint_gs:
            gaussians_copy = copy.deepcopy(gaussians)
            weak_point_gs_mask = (gaussians_copy.point_type == 0)
            strong_point_gs_mask = (gaussians_copy.point_type == 3)
            line_gs_mask = (gaussians_copy.point_type == 1)
            weak_surface_gs_mask = (gaussians_copy.point_type == 2)
            strong_surface_gs_mask = (gaussians_copy.point_type == 4)
            # only_point_gs
            final_gs = weak_point_gs_mask | strong_point_gs_mask
            final_gs_mask = ~final_gs
            gaussians_copy.prune_points(final_gs_mask)
            gaussians_copy.print_infor()
            bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            dir_flag_name = "only_point"
            if not skip_train:
                render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians_copy,
                           pipeline, background, dataset.train_test_exp, dir_flag_name)
            if not skip_test:
                render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians_copy,
                           pipeline, background, dataset.train_test_exp, dir_flag_name)


        is_only_allline_gs = False
        if is_only_allline_gs:
            gaussians_copy = copy.deepcopy(gaussians)
            weak_point_gs_mask = (gaussians_copy.point_type == 0)
            strong_point_gs_mask = (gaussians_copy.point_type == 3)
            line_gs_mask = (gaussians_copy.point_type == 1)
            weak_surface_gs_mask = (gaussians_copy.point_type == 2)
            strong_surface_gs_mask = (gaussians_copy.point_type == 4)
            # only_line_gs
            final_gs = line_gs_mask
            final_gs_mask = ~final_gs
            gaussians_copy.prune_points(final_gs_mask)
            gaussians_copy.print_infor()
            bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            dir_flag_name = "only_line"
            if not skip_train:
                render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians_copy,
                           pipeline, background, dataset.train_test_exp, dir_flag_name)
            if not skip_test:
                render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians_copy,
                           pipeline, background, dataset.train_test_exp, dir_flag_name)


        is_only_allsurface_gs = False
        if is_only_allsurface_gs:
            gaussians_copy = copy.deepcopy(gaussians)
            weak_point_gs_mask = (gaussians_copy.point_type == 0)
            strong_point_gs_mask = (gaussians_copy.point_type == 3)
            line_gs_mask = (gaussians_copy.point_type == 1)
            weak_surface_gs_mask = (gaussians_copy.point_type == 2)
            strong_surface_gs_mask = (gaussians_copy.point_type == 4)
            # only_surface_gs
            final_gs = weak_surface_gs_mask | strong_surface_gs_mask
            final_gs_mask = ~final_gs
            gaussians_copy.prune_points(final_gs_mask)
            gaussians_copy.print_infor()
            bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            dir_flag_name = "only_surface"
            if not skip_train:
                render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians_copy,
                           pipeline, background, dataset.train_test_exp, dir_flag_name)
            if not skip_test:
                render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians_copy,
                           pipeline, background, dataset.train_test_exp, dir_flag_name)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
