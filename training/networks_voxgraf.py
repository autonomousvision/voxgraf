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

import torch
from torch_utils import persistence
from training.networks_stylegan2 import SynthesisNetwork as SG2_syn, MappingNetwork as SingleMappingNetwork
from training.networks_stylegan3 import SynthesisLayer as SG3_SynthesisLayer
from training.networks_stylegan2_3d import SynthesisNetwork as Synthesis3D
from training.virtual_camera_utils import PoseSampler, Renderer

@persistence.persistent_class
class MappingNetwork(torch.nn.Module):
    """Wrapper class for fg and bg mapping network."""
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality, 0 = no labels.
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_ws_fg,                  # Number of intermediate latents to output for foreground.
        num_ws_bg,                  # Number of intermediate latents to output for background.
        use_bg=True,                # Model background with 2D GAN
        **mapping_kwargs
    ):
        super().__init__()
        self.fg = SingleMappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=num_ws_fg, **mapping_kwargs)
        self.use_bg = use_bg
        if use_bg:
            self.bg = SingleMappingNetwork(z_dim=z_dim, c_dim=0, w_dim=w_dim, num_ws=num_ws_bg, **mapping_kwargs)       # bg is not conditioned on pose

    def forward(self, z_fg, z_bg, c_fg, c_bg, **kwargs):
        w_fg = self.fg(z_fg, c_fg, **kwargs)
        w_bg = None
        if self.use_bg:
            w_bg = self.bg(z_bg, c_bg, **kwargs)
        return w_fg, w_bg


@persistence.persistent_class
class RefinementNetwork(torch.nn.Module):
    def __init__(self, w_dim, input_resolution, input_channels, output_channels, dhidden, num_layers):
        super().__init__()
        assert num_layers > 0
        last_cutoff = input_resolution / 2
        refinement = []
        for i in range(num_layers):
            is_last = (i == num_layers-1)

            refinement.append(SG3_SynthesisLayer(
                w_dim=w_dim, is_torgb=is_last, is_critically_sampled=True, use_fp16=False,
                in_channels=input_channels if i==0 else dhidden,
                out_channels=dhidden if not is_last else output_channels,
                in_size=input_resolution, out_size=input_resolution,
                in_sampling_rate=input_resolution, out_sampling_rate=input_resolution,
                in_cutoff=last_cutoff, out_cutoff=last_cutoff,
                in_half_width=0, out_half_width=0
            ))

        self.layers = torch.nn.ModuleList(refinement)

    def forward(self, x, w, update_emas=False):
        for layer in self.layers:
            x = layer(x, w, update_emas=update_emas)
        return x


@persistence.persistent_class
class SynthesisNetwork(torch.nn.Module):
    def __init__(self,
         w_dim,                 # Intermediate latent (W) dimensionality.
         img_resolution,        # Output image resolution.
         img_channels,          # Number of color channels.
         use_bg=True,           # Model background with 2D GAN
         sigma_mpl = 1,         # Rescaling factor for density
         pose_kwargs={},
         render_kwargs={},
         fg_kwargs={},
         bg_kwargs={},
         refinement_kwargs={},
    ):
        super().__init__()
        if img_channels != 3:
            raise NotImplementedError

        self.img_resolution = img_resolution
        self.use_bg = use_bg
        self.use_refinement = refinement_kwargs.get('num_layers', 0) > 0
        self.sigma_mpl = sigma_mpl

        self.pose_sampler = PoseSampler(**pose_kwargs)
        self.renderer = Renderer(**render_kwargs)

        self.generator_fg = Synthesis3D(w_dim=w_dim, img_channels=img_channels*self.renderer.basis_dim+1, architecture='skip', renderer=self.renderer, **fg_kwargs)
        if self.use_bg:
            self.generator_bg = SG2_syn(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **bg_kwargs)

        if self.use_refinement:
            self.__setattr__(f'r{self.img_resolution}', RefinementNetwork(w_dim, self.img_resolution, input_channels=3, output_channels=img_channels, **refinement_kwargs))

    def __repr__(self):
        return (
            f"SynthesisNetwork(img_resolution={self.img_resolution}, use_bg={self.use_bg}, use_refinement={self.use_refinement})"
        )

    def _get_sparse_grad_indexer(self, device):
        indexer = None#self.sparse_grad_indexer     # TODO: can we reuse it?
        if indexer is None:
            indexer = torch.empty((0,), dtype=torch.bool, device=device)
        return indexer

    def _get_sparse_sh_grad_indexer(self, device):
        indexer = None# self.sparse_sh_grad_indexer     # TODO: can we reuse it?
        if indexer is None:
            indexer = torch.empty((0,), dtype=torch.bool, device=device)
        return indexer

    def forward(self, ws, ws_bg, pose=None, noise_mode='none', update_emas=None, return_3d=False, n_views=1, render_alpha=False, render_depth=False, render_vardepth=False, raw_noise_std_sigma=0):
        B = ws.shape[0]

        # Get camera poses
        if pose is None or torch.isnan(pose).all():
            pose = torch.stack([self.pose_sampler.sample_from_poses() for _ in range(B*n_views)])

        # Generate sparse voxel grid
        density_data_list, sh_data_list, coords = self.generator_fg(ws, pose=pose, noise_mode=noise_mode, update_emas=update_emas)

        if raw_noise_std_sigma > 0:
            for i in range(B):
                density_data_list[i] = density_data_list[i] + torch.randn(density_data_list[i].shape, device=ws.device) * raw_noise_std_sigma
        if self.sigma_mpl != 1:
            for i in range(B):
                density_data_list[i] = density_data_list[i] * self.sigma_mpl

        # Render foreground
        pred = self.renderer(pose, density_data_list, sh_data_list, coords, img_resolution=self.img_resolution, grid_resolution=self.generator_fg.grid_resolution, render_alpha=(render_alpha or self.use_bg), render_depth=render_depth, render_vardepth=render_vardepth)
        pred = pred.view(B*n_views, self.img_resolution, self.img_resolution, 6).permute(0, 3, 1, 2)        # (BxHxW)xC -> # BxCxHxW
        rgb, alpha, depth, vardepth = pred[:, :3], pred[:, 3:4], pred[:, 4:5], pred[:, 5:6]

        if self.use_bg:
            rgb_bg = self.generator_bg(ws_bg, update_emas=update_emas, noise_mode=noise_mode)

            # alpha compositing
            rgb_bg = rgb_bg / 2 + 0.5  # [-1, 1] -> [0, 1]
            assert alpha.min() >= 0 and alpha.max() <= 1+1e-3             # add some offset due to precision in alpha computation
            rgb = alpha * rgb + (1 - alpha) * rgb_bg

        # [0,1] -> [-1, 1]
        rgb = rgb * 2 - 1

        # Refine composited image
        if self.use_refinement:
            w_last = ws[:, -1]          # simply reuse last w for refinement layers
            rgb = self.__getattr__(f'r{self.img_resolution}')(rgb, w_last, update_emas)

        out = {'rgb': rgb}
        if render_alpha:
            assert alpha.min() >= 0 and alpha.max() <= 1+1e-3             # add some offset due to precision in alpha computation
            out['alpha'] = alpha
        if render_depth:
            out['depth'] = depth
        if render_vardepth:
            out['vardepth'] = vardepth
        if return_3d:
            out['density'] = density_data_list
            out['sh'] = sh_data_list
            out['coords'] = coords
        return out


@persistence.persistent_class
class Generator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        use_bg              = True, # Model background with 2D GAN
        pose_conditioning   = True, # Condition generator on pose
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.pose_conditioning = pose_conditioning
        self.synthesis = SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, use_bg=use_bg, **synthesis_kwargs)
        self.num_ws = self.synthesis.generator_fg.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws_fg=self.synthesis.generator_fg.num_ws, num_ws_bg=0 if not use_bg else self.synthesis.generator_bg.num_ws, use_bg=use_bg, **mapping_kwargs)

    def forward(self, z, c, pose, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        ws, ws_bg = self.mapping(z, z, c, torch.empty_like(c[:, :0]), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        img = self.synthesis(ws, update_emas=update_emas, ws_bg=ws_bg, pose=pose, **synthesis_kwargs)
        return img

