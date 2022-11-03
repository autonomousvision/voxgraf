# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# Modified by Katja Schwarz for VoxGRAF: Fast 3D-Aware Image Synthesis with Sparse Voxel Grids
#

"""Generate lerp videos using pretrained network pickle."""

from typing import List, Optional, Tuple, Union
import os
import click
import imageio
import numpy as np
import torch
from training.virtual_camera_utils import uv2RT
from gen_images import parse_range, parse_tuple, load_model, make_inputs, generate_grids

# environment variables
os.environ['OMP_NUM_THREADS'] = "16"

#----------------------------------------------------------------------------

def sample_oval(n_points, a, b):
    "a is axes length in x-direction, b in y-direction"
    angle_range = np.linspace(0, 2*np.pi, n_points)
    x = a*np.sin(angle_range)
    y = b*np.cos(angle_range)
    return x, y

def sample_line(n_points, a):
    "a is axes length in x-direction, b in y-direction"
    angle_range = np.linspace(-1, 1, n_points)
    x = a*angle_range
    y = np.zeros(n_points)
    return x, y

def make_poses(G, label, pose_sampling, w_frames, num_keyframes=1, range_azim=(180, 90), range_polar=(180, 90)):
    assert len(label) % num_keyframes == 0
    nsamples = len(label) // num_keyframes
    # set up the camera trajectory
    if pose_sampling == 'oval':
        azim, polar = sample_oval(w_frames, range_azim[1], range_polar[1])
    elif pose_sampling == 'line':
        azim, polar = sample_line(w_frames, range_azim[1])
    else:
        raise AttributeError
    azim += range_azim[0]
    polar += range_polar[0]

    azim2u = lambda x: torch.deg2rad(torch.tensor(x)) / (2 * np.pi)
    polar2v = lambda x: 0.5 * (1 - torch.cos(torch.deg2rad(torch.tensor(x))))
    us, vs = azim2u(azim), polar2v(polar)

    RTs = []
    for u, v in zip(us, vs):
        RT = uv2RT(u, v, 1)     # radius is set later
        RTs.append(RT)
    poses = torch.stack(RTs).to(torch.float32)
    poses = poses.view(1, 1, *poses.shape).repeat(num_keyframes, nsamples, 1, 1, 1)

    if G.c_dim != 0:  # adjust the radius of the samples to match their pose
        c = label.view(num_keyframes, nsamples, 1, 3, 4).expand(-1, -1, w_frames, -1, -1)
        cradius = c[:, :, :, :3, 3].norm(dim=3, keepdim=True).min(dim=0, keepdim=True).values      # chooses smalles radius of all keyframes
        poses[:, :, :, :3, 3] *= cradius

    return poses.flatten(0, 1)      # (num_keyframes x nsamples) x wframes x 4 x 4

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds', required=True)
@click.option('--shuffle-seed', type=int, help='Random seed to use for shuffling seed order', default=None)
@click.option('--grid', type=parse_tuple, help='Grid width/height, e.g. \'4x3\' (default: 1x1)', default=(1,1))
@click.option('--num-keyframes', type=int, help='Number of seeds to interpolate through.  If not specified, determine based on the length of the seeds array given by --seeds.', default=None)
@click.option('--w-frames', type=int, help='Number of frames to interpolate between latents', default=120)      # 60*x
@click.option('--range-azim', type=parse_tuple, help='Mean and std of pose distribution azimuth angle', default=(180, 20))
@click.option('--range-polar', type=parse_tuple, help='Mean and std of pose distribution polar angle', default=(90, 10))
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--output', help='Output .mp4 filename', type=str, required=True, metavar='FILE')
@click.option('--pose-sampling', help='Camera trajectory', type=click.Choice(['oval', 'line']), default='oval', show_default=True)
@click.option('--bg-decay', help='Reduce background visibility', type=bool, default=False, show_default=True)
@click.option('--no-bg', help='Remove background', type=bool, default=False, show_default=True)

def generate_videos(
    network_pkl: str,
    seeds: List[int],
    shuffle_seed: Optional[int],
    truncation_psi: float,
    grid: Tuple[int,int],
    num_keyframes: Optional[int],
    w_frames: int,
    range_azim: Optional[int],
    range_polar: Optional[int],
    output: str,
    pose_sampling: str,
    bg_decay: bool,
    no_bg: bool,
):
    """Render a latent vector interpolation video.

    The output video length
    will be 'num_keyframes*w_frames' frames.
    """
    os.makedirs(os.path.dirname(output), exist_ok=True)
    assert len(seeds) == grid[0]*grid[1]*num_keyframes, f'need gw({grid[0]})*gh({grid[1]})*num_keyframes({num_keyframes})={grid[0]*grid[1]*num_keyframes} seeds but have {len(seeds)}'
    G = load_model(network_pkl)
    z, c = make_inputs(seeds, G)
    p = make_poses(G, c, pose_sampling, w_frames, num_keyframes=num_keyframes, range_azim=range_azim, range_polar=range_polar)

    if bg_decay:
        bg_color = 1
        n_start = w_frames // 4
        n_end = num_keyframes * w_frames - (w_frames // 4)
        get_bg_weight = lambda n: 1 - min(1, max(0, (n - n_start) / (n_end - n_start)))

    video_out = imageio.get_writer(output, mode='I', fps=60, codec='libx264', bitrate='12M')
    keyframe_idx = 0
    for ret_dict in generate_grids(G=G, truncation_psi=truncation_psi, grid=grid, latents=z, poses=p, conditions=c, ret_alpha=bg_decay, ret_depth=False, ret_bg_fg=(bg_decay or no_bg), interpolate=True):
        g = ret_dict['rgb']

        if no_bg:
            g = ret_dict['fg']
        if bg_decay:
            g_decay = []
            for i, (g_i, alpha_i) in enumerate(zip(g, ret_dict['alpha'])):
                weight = get_bg_weight(keyframe_idx*w_frames + i)
                print(weight)
                g_decay.append(alpha_i * g_i + (1 - alpha_i) * weight * g_i + (1 - alpha_i) * (1 - weight) * torch.full_like(alpha_i, fill_value=bg_color))
            g = g_decay

        for pose_idx, g_p in enumerate(g):
            video_out.append_data((g_p.permute(1,2,0)*255).to(torch.uint8).cpu().numpy())
        keyframe_idx += 1
    video_out.close()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_videos() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
