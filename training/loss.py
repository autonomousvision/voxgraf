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

"""Loss functions."""

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d
import svox2
from typing import Tuple, Optional, List


_C = svox2.utils._get_c_extension()

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------


def lerp(start, end, cur_steps, total_steps):
    if cur_steps >= total_steps:
        return end
    return cur_steps/total_steps * (end-start) + start


class TVLoss(torch.nn.Module):
    def __init__(self, grid, lambda_tv=1e-5, tv_sparsity=0.01):
        super(TVLoss, self).__init__()
        self.grid = grid

        self.lambda_tv = lambda_tv
        self.tv_sparsity = tv_sparsity
        self.tv_logalpha = False
        self.lambda_tv_lumisphere = 0.0
        self.tv_lumisphere_sparsity = 0.01
        self.tv_lumisphere_dir_factor = 0.0
        self.lambda_l2_sh = 0.0
        self.tv_contiguous = 1
        self.lambda_tv_background_sigma = 1e-2
        self.lambda_tv_background_color = 1e-2
        self.tv_background_sparsity = 0.01
        self.lambda_tv_basis = 0.0

        self.ndc_coeffs = (-1, -1)

    def forward(self, density_data_list, coords, grid_res):
        if self.lambda_tv == 0:
            return

        links = self.grid.renderer.get_links(coords, grid_res=grid_res, device=density_data_list[0].device)

        if self.lambda_tv > 0.0:
            with torch.autograd.profiler.record_function('Greg_tv_forward'):
                density_grad = self.tv_grad(density_data_list, links, grid_res,
                                            scaling=self.lambda_tv, sparse_frac=self.tv_sparsity,
                                            logalpha=self.tv_logalpha,
                                            ndc_coeffs=self.ndc_coeffs, contiguous=self.tv_contiguous)

                with torch.autograd.profiler.record_function('Greg_tv_backward'):
                    density_data = torch.cat(density_data_list)
                    density_data.backward(density_grad, retain_graph=True)

    def _get_rand_cells(self, resolution: int, sparse_frac: float, device: str, force: bool = False, contiguous:bool=True):
        if sparse_frac < 1.0 or force:
            assert self.grid.sparse_grad_indexer is None or self.grid.sparse_grad_indexer.dtype == torch.bool, \
                   "please call sparse loss after rendering and before gradient updates"
            grid_size = resolution**3
            sparse_num = max(int(sparse_frac * grid_size), 1)
            if contiguous:
                start = np.random.randint(0, grid_size)
                arr = torch.arange(start, start + sparse_num, dtype=torch.int32, device=device)

                if start > grid_size - sparse_num:
                    arr[grid_size - sparse_num - start:] -= grid_size
                return arr
            else:
                return torch.randint(0, grid_size, (sparse_num,), dtype=torch.int32, device=device)
        return None

    def tv_grad(self,
        density_data_list: List[torch.Tensor],
        links: torch.Tensor,
        resolution: int,
        scaling: float = 1.0,
        sparse_frac: float = 0.01,
        logalpha: bool = False, logalpha_delta: float = 2.0,
        ndc_coeffs: Tuple[float, float] = (-1.0, -1.0),
        contiguous: bool = True
        ):
        """
        Add gradient of total variation for sigma as in Neural Volumes
        [Lombardi et al., ToG 2019]
        directly into the gradient tensor, multiplied by 'scaling'
        """
        assert links.shape[0] == len(density_data_list)
        N, C = sum(d.shape[0] for d in density_data_list), density_data_list[0].shape[1]
        grad_all = torch.empty((N, C), dtype=density_data_list[0].data.dtype, device=density_data_list[0].data.device)
        idx_start = 0
        for density_data, links_i in zip(density_data_list, links):
            grad_holder = torch.zeros_like(density_data.data)
            assert (
                    _C is not None and density_data.is_cuda
            ), "CUDA extension is currently required for total variation"

            assert not logalpha, "No longer supported"
            rand_cells = self._get_rand_cells(resolution, sparse_frac, device=density_data.device, contiguous=contiguous)
            if rand_cells is not None:
                if rand_cells.size(0) > 0:
                    _C.tv_grad_sparse(links_i, density_data,
                                      rand_cells,
                                      self.grid._get_sparse_grad_indexer(density_data.device),
                                      0, 1, scaling,
                                      logalpha, logalpha_delta,
                                      False,
                                      self.grid.opt.last_sample_opaque,
                                      ndc_coeffs[0], ndc_coeffs[1],
                                      grad_holder)
            else:
                _C.tv_grad(links_i, density_data, 0, 1, scaling,
                           logalpha, logalpha_delta,
                           False,
                           ndc_coeffs[0], ndc_coeffs[1],
                           grad_holder)
                self.sparse_grad_indexer: Optional[torch.Tensor] = None
            
            idx_end = idx_start + len(density_data)
            grad_all[idx_start:idx_end] = grad_holder
            idx_start = idx_end
        assert idx_end == len(grad_all)
        
        return grad_all


class CoverageLoss(torch.nn.Module):
    def __init__(self, lambda_cvg_fg, min_cvg_fg, lambda_cvg_bg, min_cvg_bg):
        super(CoverageLoss, self).__init__()
        self.lambda_cvg_fg = lambda_cvg_fg
        self.min_cvg_fg = min_cvg_fg
        self.lambda_cvg_bg = lambda_cvg_bg
        self.min_cvg_bg = min_cvg_bg

    def forward(self, mask):
        assert mask.min() >=0 and mask.max() <= 1+1e-3
        assert mask.ndim == 4

        loss = 0
        if self.lambda_cvg_fg > 0:
            with torch.autograd.profiler.record_function('Greg_cvg_fg_forward'):
                fg_cvg = mask.flatten(1, -1).mean(dim=1)
                loss_fg = self.lambda_cvg_fg * (self.min_cvg_fg - fg_cvg).clamp(min=0)
                training_stats.report('Loss/G/loss_cvg_fg', loss_fg)

            loss = loss + loss_fg.sum()

        if self.lambda_cvg_bg > 0:
            with torch.autograd.profiler.record_function('Greg_cvg_bg_forward'):
                bg_cvg = (1-mask).flatten(1, -1).mean(dim=1)
                loss_bg = self.lambda_cvg_bg * (self.min_cvg_bg - bg_cvg).clamp(min=0)
                training_stats.report('Loss/G/loss_cvg_bg', loss_bg)

            loss = loss + loss_bg.sum()

        return loss


class VoxGRAFLoss(Loss):
    def __init__(self, device, G, D, augment_pipe=None, r1_gamma=10,
                 discriminator_pose_conditioning=False,
                 generator_pose_conditioning=False, p_generator_pose_conditioning=0.5, generator_pose_conditioning_steps=1e6,
                 raw_noise_std=0., decrease_noise_until=5000,
                 lambda_vardepth=0, lambda_sparsity=0, tv_kwargs={}, cvg_kwargs={},
                 no_reg_until=-1,
    ):
        super().__init__()
        self.device = device
        self.G = G
        self.D = D
        self.augment_pipe = augment_pipe
        self.r1_gamma = r1_gamma

        self.discriminator_pose_conditioning = discriminator_pose_conditioning
        self.generator_pose_conditioning = generator_pose_conditioning
        self.p_generator_pose_conditioning = p_generator_pose_conditioning
        self.generator_pose_conditioning_steps = generator_pose_conditioning_steps

        self.stats_tfevents = None         # set manually to save logs in tensorboard
        self.last_kimg = -1

        self.raw_noise_std = raw_noise_std
        self.decrease_noise_until = decrease_noise_until

        self.no_reg_until = no_reg_until
        self.tv_reg = TVLoss(grid=G.synthesis, **tv_kwargs)
        self.cvg_reg = CoverageLoss(**cvg_kwargs)
        self.lambda_vardepth = lambda_vardepth
        self.lambda_sparsity = lambda_sparsity
        self.has_regularization = self.tv_reg.lambda_tv > 0 or self.lambda_sparsity > 0    # do not count mask and depth losses here because they have to be added to loss_Gmain

    def get_raw_noise_std(self, cur_nimg):
        return lerp(self.raw_noise_std, 0, cur_nimg, self.decrease_noise_until)

    def draw_pose(self, pose, cur_nimg):
        # draw random pose with some probability >= 50% otherwise the correct pose
        assert cur_nimg is not None
        B = pose.shape[0]
        p = lerp(1, self.p_generator_pose_conditioning, cur_nimg, self.generator_pose_conditioning_steps)

        real_pose_idx = torch.arange(B)
        other_pose_idx = torch.multinomial((~torch.eye(B, dtype=torch.bool)).float(), num_samples=1).flatten()

        c_idx = torch.where(torch.rand(B) < p, other_pose_idx, real_pose_idx)
        return pose[c_idx, :3, :4].view(B, 12)

    def run_G(self, z, c, pose=None, cur_nimg=None, update_emas=False, **synthesis_kwargs):
        if self.generator_pose_conditioning:
            assert c.numel() == 0 and pose is not None
            c = self.draw_pose(pose, cur_nimg)
        raw_noise_std_sigma = self.get_raw_noise_std(cur_nimg)

        ws, ws_bg = self.G.mapping(z, z, c, torch.empty_like(c[:, :0]), update_emas=update_emas)
        img = self.G.synthesis(ws, ws_bg, pose=pose, update_emas=update_emas, raw_noise_std_sigma=raw_noise_std_sigma, **synthesis_kwargs)
        return img

    def run_D(self, img, c, pose=None, update_emas=False):
        if self.discriminator_pose_conditioning:
            assert c.numel() == 0 and pose is not None
            c = pose[:, :3, :4].flatten(1, 2)

        if self.augment_pipe is not None:
            if img.shape[1] == 4:
                img, alpha = img[:, :3], img[:, -1:]
                img = self.augment_pipe(img)
                img = torch.cat([img, alpha], 1)
            elif img.shape[1] == 6:
                img = torch.cat([self.augment_pipe(img[:,:3]), self.augment_pipe(img[:,3:])], 1)
            else:
                img = self.augment_pipe(img)

        logits = self.D(img, c, update_emas=update_emas)
        return logits

    def find_visible(self, density_data_list, sh_data_list, coords):
        # get n random views per sample
        B = len(density_data_list)
        synthesis = self.G.synthesis

        # get equally spread camera poses and rays
        pose = torch.stack(synthesis.pose_sampler.sample_on_grid((4, 4), sigma_u=3, sigma_v=3))
        pose = pose.unsqueeze(0).expand(B, -1, -1, -1).flatten(0, 1)

        return synthesis.renderer.find_visible(density_data_list, sh_data_list, coords, pose, img_resolution=synthesis.img_resolution, grid_resolution=synthesis.generator_fg.grid_resolution)

    def log_sigma_stats(self, density_data_list, cur_nimg):
        if self.stats_tfevents is not None and cur_nimg // 1000 != self.last_kimg:
            sigmas = torch.cat(density_data_list)[:, 0]
            self.stats_tfevents.add_histogram(tag='G/sigmas', values=sigmas, global_step=cur_nimg)

            alphas = 1. - torch.exp(-torch.relu(torch.cat(density_data_list)[:, 0]) * self.G.synthesis.renderer.get_delta(self.G.synthesis.generator_fg.grid_resolution))
            self.stats_tfevents.add_histogram(tag='G/alphas', values=alphas, bins=torch.linspace(0, 1, 100), global_step=cur_nimg)

            self.last_kimg = cur_nimg // 1000

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg, pose, stats_tfevents=None, **synthesis_kwargs):
        assert phase in ['Gboth', 'Dmain', 'Dreg', 'Dboth']
        if real_c.numel() > 0:
            raise NotImplementedError
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        use_regularization = self.has_regularization and cur_nimg > self.no_reg_until
        has_mask_loss = (self.cvg_reg.lambda_cvg_bg > 0 or self.cvg_reg.lambda_cvg_fg > 0) and cur_nimg > self.no_reg_until
        has_vardepth_loss = self.lambda_vardepth > 0 and cur_nimg > self.no_reg_until
        render_alpha = has_mask_loss or has_vardepth_loss

        # Gmain: Maximize logits for generated images.
        if phase == 'Gboth':
            with torch.autograd.profiler.record_function('Gmain_forward'):
                out = self.run_G(gen_z, gen_c, pose=pose, render_alpha=render_alpha, render_vardepth=has_vardepth_loss, return_3d=True, cur_nimg=cur_nimg, **synthesis_kwargs)
                gen_img = out['rgb']
                density_data_list, sh_data_list, coords = out['density'], out['sh'], out['coords']
                if has_vardepth_loss:
                    gen_vardepth = out['vardepth']
                if render_alpha:
                    gen_alpha = out['alpha']

                gen_logits = self.run_D(gen_img, gen_c, pose=pose)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                self.log_sigma_stats(density_data_list, cur_nimg)
                loss_Gmain = torch.nn.functional.softplus(-gen_logits)  # -log(sigmoid(gen_logits))
                training_stats.report('Loss/G/loss', loss_Gmain)

            # Mask loss needs to be added to loss_Gmain because we can only backprop once through the rendering
            mask_loss = 0
            if has_mask_loss:
                assert gen_alpha.ndim == 4
                # Coverage Regularization
                mask_loss = mask_loss + self.cvg_reg(gen_alpha)

            depth_loss = 0
            if has_vardepth_loss:
                if self.lambda_vardepth:
                    with torch.autograd.profiler.record_function('Greg_vardepth_forward'):
                        delta = self.G.synthesis.renderer.get_delta(self.G.synthesis.generator_fg.grid_resolution)       # distance between voxels
                        vardepth_loss = gen_vardepth.clamp(min=(3*delta)**2) * gen_alpha.detach()
                        vardepth_loss = self.lambda_vardepth * vardepth_loss.sum(dim=[1,2,3])
                    training_stats.report('Loss/G/loss_vardepth', vardepth_loss)
                    depth_loss = depth_loss + vardepth_loss.sum()

            with torch.autograd.profiler.record_function('Gmain_backward'):
                (loss_Gmain.mean().mul(gain) + mask_loss + depth_loss).backward(retain_graph=use_regularization or has_vardepth_loss)

            grid_res = self.G.synthesis.generator_fg.grid_resolution
            density = torch.cat(density_data_list)
            needs_rendered_voxels = cur_nimg % 1000 == 0

            n_dense = len(density_data_list) * grid_res ** 3
            if needs_rendered_voxels:
                rendered_voxels = self.find_visible(density_data_list, sh_data_list, coords)
                training_stats.report('Loss/frac_rendered', rendered_voxels.sum() / n_dense)        # track sparsity
            training_stats.report('Loss/frac_visible', (density > 0).sum() / n_dense)        # track visibility

            if use_regularization:
                # TV Regularization
                self.tv_reg(density_data_list, coords, grid_res)

                # Sparsity Regularization
                if self.lambda_sparsity > 0:
                    loss_sparsity = self.lambda_sparsity * (1 + 2 * density.clamp(min=0) ** 2).log().sum()
                    training_stats.report('Loss/G/loss_sparsity', loss_sparsity / len(density_data_list))
                    loss_sparsity.backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img = self.run_G(gen_z, gen_c, pose=pose, update_emas=True, cur_nimg=cur_nimg, **synthesis_kwargs)['rgb']
                gen_logits = self.run_D(gen_img, gen_c, pose=pose, update_emas=True)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits)  # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_logits = self.run_D(real_img_tmp, real_c, pose=pose)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits)  # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if phase in ['Dreg', 'Dboth']:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1, 2, 3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1).mean().mul(gain).backward()

