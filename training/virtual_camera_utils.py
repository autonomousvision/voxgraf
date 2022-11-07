import torch
import math
from svox2 import svox2              # for renderer
_C = svox2.utils._get_c_extension()


def azim2u(azim):
    """Converts azimuth angle in [0, 360] to u in [0,1]"""
    return torch.deg2rad(torch.tensor(azim))/ (2*math.pi)


def polar2v(polar):
    """Converts polar angle in [0, 180] to v in [0,1]"""
    return 0.5 * (1 - torch.cos(torch.deg2rad(torch.tensor(polar))))


def uv2sphere(u, v):
    phi = 2 * math.pi * u
    theta = torch.arccos(1 - 2 * v)
    cx = torch.sin(theta) * torch.cos(phi)
    cy = torch.sin(theta) * torch.sin(phi)
    cz = torch.cos(theta)
    s = torch.stack([cx, cy, cz])
    return s


def look_at(eye, at=torch.tensor([0, 0, 0]), up=torch.tensor([0, 0, 1]), eps=1e-5):
    at = at.to(eye.device, eye.dtype).view(1, 3)
    up = up.to(eye.device, eye.dtype).view(1, 3)

    eye = eye.view(-1, 3)
    up = up.repeat(eye.shape[0] // up.shape[0], 1)
    eps = torch.tensor([eps], device=eye.device, dtype=eye.dtype).view(1, 1).repeat(up.shape[0], 1)

    z_axis = eye - at
    z_axis /= torch.stack([z_axis.norm(dim=1, keepdim=True), eps]).max()            # not differentiable due to max

    x_axis = torch.cross(up, z_axis)
    x_axis /= torch.stack([x_axis.norm(dim=1, keepdim=True), eps]).max()

    y_axis = torch.cross(z_axis, x_axis)
    y_axis /= torch.stack([y_axis.norm(dim=1, keepdim=True), eps]).max()

    r_mat = torch.cat((x_axis.view(-1, 3, 1), y_axis.view(-1, 3, 1), z_axis.view(-1, 3, 1)), dim=2)

    return r_mat


def uv2RT(u, v, radius):
    T = radius * uv2sphere(u, v)
    R = look_at(T)[0]

    T = T.view(3, 1)
    RT = torch.cat([R, T], dim=1)
    RT = torch.cat([RT, torch.tensor([0, 0, 0, 1], dtype=RT.dtype, device=RT.device).view(1, 4)])

    RT = opengl2opencv(RT)

    return RT


def opengl2opencv(C2W):
    flip_yz = torch.eye(4, dtype=C2W.dtype, device=C2W.device)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    C2W = torch.matmul(C2W, flip_yz)
    return C2W


class UVSampler(torch.nn.Module):
    def __init__(self):
        super(UVSampler, self).__init__()
        self.dist_u = None
        self.dist_v = None

    def sample_on_grid(self, gridsize, **kwargs):
        raise NotImplementedError


class UniformUVSampler(UVSampler):
    def __init__(self, range_azim, range_polar):
        super(UniformUVSampler, self).__init__()

        u_min, u_max = azim2u(range_azim)
        v_min, v_max = polar2v(range_polar)

        self.dist_u = torch.distributions.uniform.Uniform(low=u_min, high=u_max)
        self.dist_v = torch.distributions.uniform.Uniform(low=v_min, high=v_max)

    def _apply(self, fn):
        super(UniformUVSampler, self)._apply(fn)
        self.dist_u.low = fn(self.dist_u.low)
        self.dist_v.low = fn(self.dist_v.low)

    def sample_on_grid(self, gridsize, **kwargs):
        nu, nv = gridsize
        if nu == 1:     # use mean
            us = torch.tensor([0.5 * (self.dist_u.low + self.dist_u.high)], device=self.dist_u.low.device)
        else:
            us = torch.linspace(self.dist_u.low, self.dist_u.high, nu, device=self.dist_u.low.device)

        if nv == 1:     # use mean
            vs = torch.tensor([0.5 * (self.dist_v.low + self.dist_v.high)], device=self.dist_v.low.device)
        else:
            vs = torch.linspace(self.dist_v.low, self.dist_v.high, nv, device=self.dist_v.low.device)

        return us, vs


class NormalUVSampler(UVSampler):
    def __init__(self, range_azim, range_polar):
        super(NormalUVSampler, self).__init__()

        u_mean, u_std = azim2u(range_azim)  # linear transform
        v_mean = polar2v(range_polar[0])
        v_high = range_polar[0] + range_polar[1]
        v_low = range_polar[0] - range_polar[1]
        v_std = max(polar2v(v_high) - v_mean, v_mean - polar2v(v_low))

        self.dist_u = torch.distributions.normal.Normal(loc=u_mean, scale=u_std)
        self.dist_v = torch.distributions.normal.Normal(loc=v_mean, scale=v_std)

    def _apply(self, fn):
        super(NormalUVSampler, self)._apply(fn)
        self.dist_u.loc = fn(self.dist_u.loc)
        self.dist_v.loc = fn(self.dist_v.loc)
        self.dist_u.scale = fn(self.dist_u.scale)
        self.dist_v.scale = fn(self.dist_v.scale)

    def sample_on_grid(self, gridsize, sigma_u=1, sigma_v=1):
        nu, nv = gridsize
        mean_u, std_u = self.dist_u.loc, sigma_u * self.dist_u.scale
        us = torch.linspace(mean_u - std_u, mean_u + std_u, nu, device=mean_u.device)

        mean_v, std_v = self.dist_v.loc, sigma_v * self.dist_v.scale
        vs = torch.linspace(mean_v - std_v, mean_v + std_v, nv, device=mean_v.device)

        return us, vs


class PoseSampler(torch.nn.Module):
    def __init__(self, range_azim, range_polar, radius, dist='uniform', poses=None):
        super(PoseSampler, self).__init__()
        self.eps = 1e-6     # exclude poles for polar angle
        self.register_buffer('radius', torch.tensor(radius))
        self.register_buffer('poses', poses)

        if dist == 'uniform':
            self.uv_sampler = UniformUVSampler(range_azim, range_polar)
        elif dist == 'normal':
            self.uv_sampler = NormalUVSampler(range_azim, range_polar)
        else:
            raise AttributeError(f'Unknown dist {dist}')

    def sample_from_poses(self):
        assert self.poses is not None
        return self.poses[torch.randint(len(self.poses), (1,))][0]

    def sample_from_dist(self):
        # sample location on unit sphere
        u = self.uv_sampler.dist_u.sample()
        v = self.uv_sampler.dist_v.sample()

        # ensure to stay within valid range
        u = u.clamp(0, 1)
        v = v.clamp(self.eps, 1-self.eps)

        RT = uv2RT(u, v, self.radius)

        return RT

    def sample_on_grid(self, gridsize, **kwargs):
        """Sample camera poses on grid."""
        us, vs = self.uv_sampler.sample_on_grid(gridsize, **kwargs)
        vs = vs.clamp(self.eps)

        RTs = []
        for u in us:
            for v in vs:
                RT = uv2RT(u, v, self.radius)
                RTs.append(RT)

        return RTs


class RaySampler(object):
    def __init__(self, img_resolution, fov):
        super(RaySampler, self).__init__()
        self.N_samples = img_resolution**2
        self.scale = torch.ones(1,).float()
        self.img_resolution = img_resolution
        self.fov = fov
        self.focal = self.res2focal(self.img_resolution, self.fov)

    @staticmethod
    def res2focal(res, fov):
        return res/2 * 1 / math.tan((.5 * fov * math.pi/180.))

    def __call__(self, pose):
        assert pose.shape[-2:] == torch.Size([3, 4])
        rays_o, rays_d = self.get_rays(self.img_resolution, self.img_resolution, self.focal, pose)
        return rays_o, rays_d

    @staticmethod
    def sample_at(pose, resolution, fov):
        assert pose.shape[-2:] == torch.Size([3, 4])
        focal = RaySampler.res2focal(resolution, fov)
        rays_o, rays_d = RaySampler.get_rays(resolution, resolution, focal, pose)
        return rays_o, rays_d

    def sample_rays(self):
        return torch.arange(0, self.img_resolution*self.img_resolution)

    # Ray helpers
    @staticmethod
    def get_rays(H, W, focal, c2w):
        # Generate rays
        yy, xx = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=c2w.device) + 0.5,             # shift by 0.5
            torch.arange(W, dtype=torch.float32, device=c2w.device) + 0.5,
        indexing='ij')

        cx = W/2
        cy = H/2

        xx = (xx - cx) / focal
        yy = (yy - cy) / focal
        zz = torch.ones_like(xx)
        dirs = torch.stack((xx, yy, zz), dim=-1)  # OpenCV convention
        dirs /= torch.norm(dirs, dim=-1, keepdim=True)
        dirs = dirs.reshape(1, -1, 3, 1)
        del xx, yy, zz

        dirs = (c2w[:, None, :3, :3] @ dirs)[..., 0]
        origins = c2w[:, None, :3, 3].expand(-1, H * W, -1).contiguous()

        return origins.view(-1, 3), dirs.view(-1, 3)


class Renderer(torch.nn.Module):
    def __init__(self, basis_type, basis_dim, opt_kwargs, grid_radius, grid_center, fov):
        super(Renderer, self).__init__()
        self.basis_type = basis_type
        if self.basis_type != svox2.BASIS_TYPE_SH:
            raise NotImplementedError
        self.basis_type_name = 'SH'
        self.basis_dim = basis_dim
        self.opt = svox2.RenderOptions(**opt_kwargs)
        self.grid_radius = torch.tensor(grid_radius, dtype=torch.float)
        self.grid_center = torch.tensor(grid_center, dtype=torch.float)
        self._offset = 0.5 * (1.0 - self.grid_center / self.grid_radius)
        self._scaling = 0.5 / self.grid_radius
        self.fov = fov

    def get_delta(self, grid_resolution):
        return (self.opt.step_size / (torch.tensor(grid_resolution) * self._scaling)).max()  # distance between voxels in world scale

    def _to_cpp(self, density_data: torch.FloatTensor, sh_data: torch.FloatTensor, links: torch.LongTensor):
        """
        Generate object to pass to C++
        """
        gspec = _C.SparseGridSpec()
        gspec.density_data = density_data
        gspec.sh_data = sh_data
        gspec.links = links

        gsz = torch.tensor(links.shape, device="cpu", dtype=torch.float32)
        gspec._offset = self._offset * gsz - 0.5
        gspec._scaling = self._scaling * gsz

        gspec.basis_dim = self.basis_dim
        gspec.basis_type = self.basis_type
        return gspec

    def _fetch_links(self, links, density_data, sh_data):
        """Overwrite in-place version because we have different links for different instances."""
        results_sigma = torch.zeros((links.size(0), 1), device=links.device, dtype=torch.float32)
        results_sh = torch.zeros((links.size(0), sh_data.size(1)), device=links.device, dtype=torch.float32)

        mask = links >= 0
        idcs = links[mask].long()

        results_sigma[mask] = density_data[idcs]
        results_sh[mask] = sh_data[idcs]
        return results_sigma, results_sh

    def _volume_render_gradcheck_lerp(self, rays: svox2.Rays, links, density_data, sh_data, render_vardepth: bool=True):
        """
        trilerp gradcheck version. Overwrite in-place version because we have different links for different instances.
        """
        # Get expected depth for vardepth computation
        if render_vardepth:
            expected_depth = self._volume_render_gradcheck_lerp(rays, links, density_data, sh_data, render_vardepth=False)[:, -2]

        B = rays.dirs.size(0)
        assert rays.origins.size(0) == B
        device = rays.origins.device
        gsz = torch.tensor(links.shape, device=device)

        # convert origins from world to grid scale
        offset = self._offset.to(device) * gsz - 0.5
        scaling = self._scaling.to(device) * gsz
        origins = torch.addcmul(offset, rays.origins, scaling)

        dirs = rays.dirs / torch.norm(rays.dirs, dim=-1, keepdim=True)
        viewdirs = dirs
        dirs = dirs * scaling
        delta_scale = 1.0 / dirs.norm(dim=1)
        dirs *= delta_scale.unsqueeze(-1)

        sh_mult = svox2.utils.eval_sh_bases(self.basis_dim, viewdirs)
        invdirs = 1.0 / dirs

        t1 = (-0.5 - origins) * invdirs
        t2 = (gsz - 0.5 - origins) * invdirs

        t = torch.min(t1, t2)
        t[dirs == 0] = -1e9
        t = torch.max(t, dim=-1).values.clamp_min_(self.opt.near_clip)

        tmax = torch.max(t1, t2)
        tmax[dirs == 0] = 1e9
        tmax = torch.min(tmax, dim=-1).values

        log_light_intensity = torch.zeros(B, device=origins.device)
        cout = sh_data.shape[1] // self.basis_dim
        out_rgb = torch.zeros((B, cout), device=origins.device)
        out_alpha = torch.zeros((B, 1), device=origins.device)
        out_depth = torch.zeros((B, 1), device=origins.device)
        out_vardepth = torch.zeros((B, 1), device=origins.device)
        good_indices = torch.arange(B, device=origins.device)

        origins_ini = origins
        dirs_ini = dirs

        mask = t <= tmax
        good_indices = good_indices[mask]
        origins = origins[mask]
        dirs = dirs[mask]

        #  invdirs = invdirs[mask]
        del invdirs
        t = t[mask]
        sh_mult = sh_mult[mask]
        tmax = tmax[mask]

        while good_indices.numel() > 0:
            pos = origins + t[:, None] * dirs
            pos = pos.clamp_min_(0.0)
            pos[:, 0] = torch.clamp_max(pos[:, 0], gsz[0].item() - 1)
            pos[:, 1] = torch.clamp_max(pos[:, 1], gsz[1].item() - 1)
            pos[:, 2] = torch.clamp_max(pos[:, 2], gsz[2].item() - 1)
            #  print('pym', pos, log_light_intensity)

            l = pos.to(torch.long)
            l.clamp_min_(0)
            l[:, 0] = torch.clamp_max(l[:, 0], gsz[0].item() - 2)
            l[:, 1] = torch.clamp_max(l[:, 1], gsz[1].item() - 2)
            l[:, 2] = torch.clamp_max(l[:, 2], gsz[2].item() - 2)
            pos -= l

            # BEGIN CRAZY TRILERP
            lx, ly, lz = l.unbind(-1)
            links000 = links[lx, ly, lz]
            links001 = links[lx, ly, lz + 1]
            links010 = links[lx, ly + 1, lz]
            links011 = links[lx, ly + 1, lz + 1]
            links100 = links[lx + 1, ly, lz]
            links101 = links[lx + 1, ly, lz + 1]
            links110 = links[lx + 1, ly + 1, lz]
            links111 = links[lx + 1, ly + 1, lz + 1]

            sigma000, rgb000 = self._fetch_links(links000, density_data, sh_data)
            sigma001, rgb001 = self._fetch_links(links001, density_data, sh_data)
            sigma010, rgb010 = self._fetch_links(links010, density_data, sh_data)
            sigma011, rgb011 = self._fetch_links(links011, density_data, sh_data)
            sigma100, rgb100 = self._fetch_links(links100, density_data, sh_data)
            sigma101, rgb101 = self._fetch_links(links101, density_data, sh_data)
            sigma110, rgb110 = self._fetch_links(links110, density_data, sh_data)
            sigma111, rgb111 = self._fetch_links(links111, density_data, sh_data)

            wa, wb = 1.0 - pos, pos
            c00 = sigma000 * wa[:, 2:] + sigma001 * wb[:, 2:]
            c01 = sigma010 * wa[:, 2:] + sigma011 * wb[:, 2:]
            c10 = sigma100 * wa[:, 2:] + sigma101 * wb[:, 2:]
            c11 = sigma110 * wa[:, 2:] + sigma111 * wb[:, 2:]
            c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
            c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]
            sigma = c0 * wa[:, :1] + c1 * wb[:, :1]

            c00 = rgb000 * wa[:, 2:] + rgb001 * wb[:, 2:]
            c01 = rgb010 * wa[:, 2:] + rgb011 * wb[:, 2:]
            c10 = rgb100 * wa[:, 2:] + rgb101 * wb[:, 2:]
            c11 = rgb110 * wa[:, 2:] + rgb111 * wb[:, 2:]
            c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
            c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]
            rgb = c0 * wa[:, :1] + c1 * wb[:, :1]

            # END CRAZY TRILERP

            log_att = (
                -self.opt.step_size
                * torch.relu(sigma[..., 0])
                * delta_scale[good_indices]
            )
            weight = torch.exp(log_light_intensity[good_indices]) * (
                1.0 - torch.exp(log_att)
            )
            # [B', 3, n_sh_coeffs]
            rgb_sh = rgb.reshape(-1, cout, self.basis_dim)
            rgb = torch.clamp_min(
                torch.sum(sh_mult.unsqueeze(-2) * rgb_sh, dim=-1) + 0.5,
                0.0,
            )  # [B', 3]
            rgb = weight[:, None] * rgb[:, :cout]

            out_rgb[good_indices] += rgb
            out_alpha[good_indices] += weight[:, None]
            out_depth[good_indices] += weight[:, None] * t[:, None].clone() * delta_scale[good_indices, None]
            if render_vardepth > 0:
                out_vardepth[good_indices] += weight[:, None] * (expected_depth[good_indices, None] - t[:, None].clone() * delta_scale[good_indices, None])**2
            log_light_intensity[good_indices] += log_att
            t += self.opt.step_size

            mask = t <= tmax
            good_indices = good_indices[mask]
            origins = origins[mask]
            dirs = dirs[mask]
            #  invdirs = invdirs[mask]
            t = t[mask]
            sh_mult = sh_mult[mask]
            tmax = tmax[mask]

        # Normalize depth and vardepth
        out_depth = out_depth / out_alpha.clamp(min=1e-10)
        out_vardepth = out_vardepth / out_alpha.clamp(min=1e-10)

        # Add background color
        if self.opt.background_brightness:
            out_rgb += (
                torch.exp(log_light_intensity).unsqueeze(-1)
                * self.opt.background_brightness
            )

        return torch.cat([out_rgb, out_alpha, out_depth, out_vardepth], 1)

    @staticmethod
    def get_links(coords, grid_res, device):
        B = coords[:, 0].max() + 1
        assert coords[:, 1:].max() < grid_res
        n_per_elem = [(coords[:, 0] == i).sum() for i in range(B)]

        idcs = torch.cat([torch.arange(n, dtype=torch.int32, device=device) for n in n_per_elem])
        links = torch.full((B, grid_res, grid_res, grid_res), -1, dtype=torch.int32, device=device)
        links[coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]] = idcs
        return links

    def forward(
            self, pose: torch.FloatTensor, density_data: torch.FloatTensor, sh_data: torch.FloatTensor, coords: torch.LongTensor,
            img_resolution: int, grid_resolution: int,
            render_alpha: bool=False, render_depth: bool=False, render_vardepth: bool=False,
            use_kernel: bool=True
    ):
        """
        ! Adapted from plenoxels: https://github.com/sxyu/svox2 !
        :param rays: Rays, (origins (N, 3), dirs (N, 3))
        :param density_data, [B, (Nvox, 1)]
        :param sh_data, [B, (Nvox, 1)], Nvox may be vary within one batch?
        :param coords, [B, (Nvox, 3)]
        :param use_kernel: bool, if false uses pure PyTorch version even if on CUDA.
        :param randomize: bool, whether to enable randomness
        :param return_raylen: bool, if true then only returns the length of the
                                    ray-cube intersection and quits
        :return: (N, 3), predicted RGB
        """
        assert all(x.ndim == 2 and x.shape[1] == 1 for x in density_data)
        assert all(x.ndim == 2 for x in sh_data)
        assert coords.ndim == 2 and coords.shape[1] == 4  # Nx4, [Nx0] -> batch index, [Nx1:3] -> 3D coords
        B = len(density_data)
        assert len(sh_data) == B
        assert coords[:, 0].max().item() == B - 1

        # Ray Sampling
        n_views = len(pose) / B
        assert n_views % 1 == 0, f'len(poses) needs to be multiple of {B} but is {len(pose)}'
        n_views = int(n_views)
        rays_o, rays_d = RaySampler.sample_at(pose[..., :3, :4], img_resolution, self.fov)
        rays_o, rays_d = rays_o.view(B, n_views*img_resolution**2, 3), rays_d.view(B, n_views*img_resolution**2, 3)
        assert rays_o.ndim == 3 and rays_o.shape[2] == 3
        assert rays_d.ndim == 3 and rays_d.shape[2] == 3
        assert (rays_o.shape[0] == B) and (rays_d.shape[0] == B)

        # Volume Rendering
        links = self.get_links(coords, grid_resolution, device=rays_o.device)

        cout = 3+3        # RGB + Alpha + Depth + Vardepth
        rgb_pred = torch.empty((B, rays_o.shape[1], cout), device=density_data[0].device)
        for i, (density_data_i, sh_data_i, links_i, rays_o_i, rays_d_i) in enumerate(zip(density_data, sh_data, links, rays_o, rays_d)):
            rays = svox2.Rays(rays_o_i, rays_d_i)

            if use_kernel and rays_o.is_cuda and rays_d.is_cuda and _C is not None:
                rgb_i, alpha_i, depth_i, vardepth_i = svox2._VolumeRenderFunction.apply(
                    density_data_i.to(torch.float32),
                    sh_data_i.to(torch.float32),
                    None,  # basis_data
                    None,
                    self._to_cpp(density_data_i.to(torch.float32), sh_data_i.to(torch.float32), links_i),
                    rays._to_cpp(),
                    self.opt._to_cpp(),
                    self.opt.backend,
                    render_alpha,
                    render_depth,
                    render_vardepth,
                )
            else:
                rend = self._volume_render_gradcheck_lerp(rays, links_i, density_data_i, sh_data_i)
                rgb_i, alpha_i, depth_i, vardepth_i = rend[:, :-3], rend[:, -3], rend[:, -2], rend[:, -1]

            if not render_alpha:
                alpha_i = torch.zeros_like(rgb_i[:, 0]).to(density_data_i.dtype)

            if not render_depth:
                depth_i = torch.zeros_like(rgb_i[:, 0]).to(density_data_i.dtype)

            if not render_vardepth:
                vardepth_i = torch.zeros_like(rgb_i[:, 0]).to(density_data_i.dtype)

            rgb_pred[i] = torch.cat([rgb_i, alpha_i[:, None], depth_i[:, None], vardepth_i[:, None]], 1).to(density_data_i.dtype)

        return rgb_pred

    def find_visible(self, density_data_list, sh_data_list, coords, poses, img_resolution, grid_resolution):
        # detach data from 3D generator graph
        density_data_list = [d.detach().requires_grad_(True) for d in density_data_list]
        sh_data_list = [s.detach().requires_grad_(True) for s in sh_data_list]

        alpha = self.forward(poses, density_data_list, sh_data_list, coords, img_resolution, grid_resolution, render_alpha=True)[..., 3:4]

        grads = torch.autograd.grad(outputs=[alpha.sum()], inputs=density_data_list)
        is_visible = (torch.cat(grads) != 0) & (torch.cat(density_data_list) > 0)
        return is_visible

    def extra_repr(self):
        return ' '.join([
            f"(basis_type={self.basis_type_name}, "
            f"basis_dim={self.basis_dim})"])