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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from time import time as get_time

############################################################################



#########################################################################

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, stage="fine", cam_type=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    
    means3D = pc.get_xyz
    if cam_type != "PanopticSports":
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=pipe.debug
        )
        time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
    else:
        raster_settings = viewpoint_camera['camera']
        time=torch.tensor(viewpoint_camera['time']).to(means3D.device).repeat(means3D.shape[0],1)
        

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # means3D = pc.get_xyz
    # add deformation to each points
    # deformation = pc.get_deformation

    
    means2D = screenspace_points
    opacity = pc._opacity

    shs, reflectance = pc.get_features

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    ref = None  # reflectance
    dark_image = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling
        rotations = pc._rotation
    deformation_point = pc._deformation_table


    if "coarse" in stage: 
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D, scales, rotations, opacity, shs

    elif "fine" in stage:
        # time0 = get_time()
        # means3D_deform, scales_deform, rotations_deform, opacity_deform = pc._deformation(means3D[deformation_point], scales[deformation_point], 
        #                                                                  rotations[deformation_point], opacity[deformation_point],
        #                                                                  time[deformation_point])

        means3D_final, scales_final, rotations_final, opacity_final, shs_final, shs_deform  = pc._deformation(means3D, scales, 
                                                                 rotations, opacity, shs,
                                                                 time)

        shs_final = shs_final * shs_deform.unsqueeze(-1)


    else:
        raise NotImplementedError

    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity_final)


    shs_final = shs_final.permute(0,2,1)
    dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(shs_final.shape[0], 1))
    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    sh2rgb = eval_sh(pc.active_sh_degree, shs_final, dir_pp_normalized)

    #sh2rgb = sh2rgb * reflectance.squeeze(dim=1)       #### view dependent illuminace * view independent reflectance
    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    colors_precomp = colors_precomp[:,0].unsqueeze(1)
    colors_precomp = torch.cat([colors_precomp, reflectance.squeeze(dim=1)],dim=-1)

    rendered_image, radii, depth = rasterizer(
        means3D = means3D_final,
        means2D = means2D,
        shs = None,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = cov3D_precomp)


    ill = rendered_image[0,:,:]
    ref = rendered_image[1:4,:,:]


    gamma_enh = pc.light2rgb(rendered_image[4:,:,:].unsqueeze(0))
    gamma1 = gamma_enh[:,0,:,:] 
    gamma2 = gamma_enh[:,1,:,:]


    ill = torch.clamp(ill,0.0,1.0)
    ref = torch.clamp(ref,0.0,1.0)


    well_image = ref*ill
    dark_image = (ill**gamma2)*(ref**gamma1)

    dark_image = torch.clamp(dark_image,0.0,1.0).squeeze(0)
    well_image = torch.clamp(well_image, 0.0, 1.0)

    return {"render":well_image,
            "dark_image": dark_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            'illumination':ill.unsqueeze(0),
            'reflectance': ref,
            'gamma1': gamma1,
            'gamma2': gamma2,
            'depth': depth,
            }


