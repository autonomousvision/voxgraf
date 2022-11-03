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

"""Generate images using pretrained network pickle."""

import os
import re
from typing import List, Optional, Tuple, Union
from tqdm import tqdm
import scipy
import click
import dnnlib
import numpy as np
import torch
from training.training_loop import depth_to_color
from torchvision.utils import make_grid, save_image

import legacy

# environment variables
os.environ['OMP_NUM_THREADS'] = "16"

#----------------------------------------------------------------------------

def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')

#----------------------------------------------------------------------------

def parse_tuple(s: Union[str, Tuple[int,int]]) -> Tuple[int, int]:
    '''Parse a 'M,N' or 'MxN' integer tuple.

    Example:
        '4x2' returns (4,2)
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    m = re.match(r'^(\d+)[x,](\d+)$', s)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    raise ValueError(f'cannot parse tuple {s}')

#----------------------------------------------------------------------------

def load_model(network_pkl, device='cuda'):
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore
        # prune last resolution to be faster
        G.synthesis.generator_fg.pruned_resolutions.append(G.synthesis.generator_fg.block_resolutions[-1])
    return G

def make_inputs(seeds, G):
    z = torch.cat([torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)) for seed in seeds])

    # Labels.
    label = torch.zeros([1, G.c_dim])
    if G.c_dim != 0:
        # sample poses for conditioning G
        label = []
        for seed in seeds:
            torch.random.manual_seed(seed)
            label.append(G.synthesis.pose_sampler.sample_from_poses()[:3, :4].flatten())
        label = torch.stack(label).to('cpu')

    return z, label

def make_poses(G, label, pose_sampling, npose):
    nsamples = len(label)
    if pose_sampling == 'random':
        poses = torch.stack([G.synthesis.pose_sampler.sample_from_poses() for _ in range(nsamples * npose)]).to('cpu')
        poses = poses.unflatten(0, (nsamples, npose))
        if G.c_dim != 0:  # adjust the radius of the samples to match their pose
            c = label.view(-1, 1, 3, 4).expand(-1, npose, -1, -1)
            cradius = c[:, :, :3, 3].norm(dim=2, keepdim=True)
            poses[:, :, :3, 3] = poses[:, :, :3, 3] / G.synthesis.pose_sampler.radius.cpu() * cradius
    elif pose_sampling == 'cond':
        assert npose == 1
        assert G.c_dim != 0
        poses = label.view(nsamples, npose, 3, 4)
    elif pose_sampling == 'equidistant':
        poses = torch.stack(G.synthesis.pose_sampler.sample_on_grid((npose, 1), sigma_u=1, sigma_v=1)).to('cpu')
        poses = poses.unsqueeze(0).expand(nsamples, -1, -1, -1)
    else:
        raise AttributeError
    return poses

def generate_grids(
    G,
    truncation_psi: float,
    grid: Tuple[int, int],
    latents: torch.Tensor,
    poses: torch.Tensor,
    conditions: torch.Tensor,
    ret_alpha: bool,
    ret_depth: bool,
    ret_bg_fg: bool,
    device='cuda',
    interpolate=False,
):
    nsamples = latents.shape[0]
    assert (conditions.shape[0] == nsamples) and (poses.shape[0] == nsamples)
    npose = poses.shape[1]

    grid_w, grid_h = grid
    if nsamples % (grid_w*grid_h) != 0:
        raise ValueError('Number of samples must be divisible by grid W*H')

    # Generate images.
    out = {'rgb': [], 'alpha': [], 'depth': [], 'fg': [], 'bg': []}
    for sample_idx, (z, c, p) in tqdm(enumerate(zip(latents, conditions, poses)), desc='iterate over samples', total=nsamples):
        z = z.to(device).unsqueeze(0)
        c = c.to(device).unsqueeze(0)
        c_bg = torch.empty_like(c[:, :0])
        p = p.to(device).unsqueeze(0)

        ws, ws_bg = G.mapping(z_fg=z, z_bg=z, c_fg=c, c_bg=c_bg, truncation_psi=truncation_psi)

        # create interpolation if needed
        interp = None
        interp_bg = None
        if interpolate:
            wraps = 1
            idx_next = (sample_idx + grid_w * grid_h) % nsamples        # same position on next grid, i.e. sample to interpolate to, for last grid go back to first
            z_next = latents[idx_next].to(device).unsqueeze(0)
            c_next = conditions[idx_next].to(device).unsqueeze(0)
            c_bg_next = torch.empty_like(c_next[:, :0])
            ws_next, ws_bg_next = G.mapping(z_fg=z_next, z_bg=z_next, c_fg=c_next, c_bg=c_bg_next, truncation_psi=truncation_psi)
            x = np.arange(-2*wraps, (2*(wraps + 1)))
            y = np.tile(torch.cat([ws, ws_next]).cpu().numpy(), [wraps * 2 + 1, 1, 1])
            interp = scipy.interpolate.interp1d(x, y, kind='cubic', axis=0)
            y_bg = np.tile(torch.cat([ws_bg, ws_bg_next]).cpu().numpy(), [wraps * 2 + 1, 1, 1])
            interp_bg = scipy.interpolate.interp1d(x, y_bg, kind='cubic', axis=0)

        # iterate over poses
        for pose_idx in tqdm(range(npose), leave=False, desc='iterate over poses'):
            p_i = p[:, pose_idx]

            # interpolate latent if needed
            if interp is not None:
                ws = torch.from_numpy(interp(pose_idx / npose)).to(device).unsqueeze(0)
            if interp_bg is not None:
                ws_bg = torch.from_numpy(interp_bg(pose_idx / npose)).to(device).unsqueeze(0)

            ret_dict = G.synthesis(ws, ws_bg, pose=p_i, noise_mode='none', render_alpha=ret_alpha or ret_bg_fg, render_depth=ret_depth)
            for k, v in ret_dict.items():
                ret_dict[k] = v.detach().cpu()

            out['rgb'].append(ret_dict['rgb'])
            if ret_depth:
                depth = depth_to_color(ret_dict['depth'], vmin=(G.synthesis.pose_sampler.radius - G.synthesis.renderer.grid_radius.norm()).item(), vmax=(G.synthesis.pose_sampler.radius + G.synthesis.renderer.grid_radius.norm()).item(), cmap_name='inferno_r')
                out['depth'].append(torch.from_numpy(depth) * 2 - 1)
            if ret_alpha or ret_bg_fg:
                if ret_alpha:
                    out['alpha'].append(ret_dict['alpha'] * 2 - 1)

                if ret_bg_fg:
                    bg_color = 1
                    alpha = ret_dict['alpha']
                    rgb = ret_dict['rgb'] / 2 + 0.5
                    out['fg'].append((alpha * rgb + (1-alpha) * bg_color) * 2 - 1)

                    # get background
                    if pose_idx == 0:
                        if G.synthesis.use_bg:
                            c_bg = torch.empty_like(c[:, :0])
                            ws, ws_bg = G.mapping(z, z, c, c_bg, truncation_psi=truncation_psi)
                            bg_img = G.synthesis.generator_bg(ws_bg, noise_mode='none')
                            if G.synthesis.use_refinement:
                                w_last = ws[:, -1]
                                refinement_net = G.synthesis.__getattr__(f'r{G.synthesis.img_resolution}')
                                bg_img = refinement_net(bg_img, w_last)
                            out['bg'].append(bg_img)

        # flush grid
        grids = {k: [] for k, v in out.items() if len(v) > 0}
        if ((sample_idx+1) % (grid_h*grid_w)) == 0:
            for k, v in out.items():
                if len(v) == 0: continue
                v = torch.cat(v).unflatten(0, (grid_h*grid_w, -1))
                for pose_idx in range(v.shape[1]):
                    imgrid = make_grid(v[:, pose_idx], nrow=grid_h, padding=0, normalize=True, value_range=(-1, 1))
                    grids[k].append(imgrid)

            yield grids
            out = {'rgb': [], 'alpha': [], 'depth': [], 'fg': [], 'bg': []}

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--grid', type=parse_tuple, help='Grid width/height, e.g. \'4x3\' (default: 1x1)', default=(1,1))
@click.option('--npose', type=int, help='Number of poses rendered for each image', default=1, show_default=True)
@click.option('--pose-sampling', type=click.Choice(['equidistant', 'random', 'cond', 'video']), default='random', show_default=True)
@click.option('--save_alpha', type=bool, default=False, show_default=True)
@click.option('--save_depth', type=bool, default=False, show_default=True)
@click.option('--save_bg_fg', type=bool, default=False, show_default=True)
@click.option('--save_poses', type=bool, default=False, show_default=True)
@click.option('--save_conditions', type=bool, default=False, show_default=True)

def generate_images(
    network_pkl: str,
    seeds: List[int],
    truncation_psi: float,
    outdir: str,
    grid: Tuple[int, int],
    npose: int,
    pose_sampling: str,
    save_alpha: bool,
    save_depth: bool,
    save_bg_fg: bool,
    save_poses: bool,
    save_conditions: bool,
):
    os.makedirs(outdir, exist_ok=True)

    G = load_model(network_pkl)
    z, c = make_inputs(seeds, G)
    p = make_poses(G, c, pose_sampling, npose)

    if save_poses:
        for sample_idx, seed in enumerate(seeds):
            for pose_idx in range(npose):
                with open(f'{outdir}/seed{seed:04d}_{pose_idx:04d}_pose.npy', 'wb') as f:
                    np.save(f, p[sample_idx:sample_idx + 1, pose_idx].cpu().numpy())

    if save_conditions:
        for sample_idx, seed in enumerate(seeds):
            with open(f'{outdir}/seed{seed:04d}_{pose_idx:04d}_label.npy', 'wb') as f:
                np.save(f, c[sample_idx:sample_idx + 1].cpu().numpy())

    nsamples = 0
    for ret_dict in generate_grids(G=G, truncation_psi=truncation_psi, grid=grid, latents=z, poses=p, conditions=c, ret_alpha=save_alpha, ret_depth=save_depth, ret_bg_fg=save_bg_fg):
        ret_seeds = seeds[nsamples:nsamples+grid[0]*grid[1]]
        for k, g in ret_dict.items():
            for pose_idx, g_p in enumerate(g):
                fname = f'{outdir}/{k}_{ret_seeds[0]:04d}_{ret_seeds[-1]:04d}.png'
                if len(g) > 1:
                    fname = fname.replace('.png', f'_{pose_idx:03d}.png')
                save_image(g, fname)
        nsamples += grid[0]*grid[1]


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
