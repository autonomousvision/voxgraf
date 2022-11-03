import svox2
import torch
import torch.nn.functional as F
from svox2.utils import Timing

torch.random.manual_seed(2)
#  torch.random.manual_seed(8289)

device = 'cuda:0'
torch.cuda.set_device(device)
dtype = torch.float32
grid = svox2.SparseGrid(
                     reso=128,
                     center=[0.0, 0.0, 0.0],
                     radius=[1.0, 1.0, 1.0],
                     basis_dim=9,
                     use_z_order=True,
                     device=device,
                     background_nlayers=0,
                     basis_type=svox2.BASIS_TYPE_SH)
grid.opt.backend = 'cuvol'
grid.opt.sigma_thresh = 0.0
grid.opt.stop_thresh = 0.0
grid.opt.background_brightness = 0.0

print(grid.sh_data.shape)

# Render using a white color to obtain equivalent to alpha rendering
from svox2.utils import SH_C0
grid.sh_data.data = torch.zeros_like(grid.sh_data.data.view(-1, 3, grid.basis_dim))
grid.sh_data.data[..., 0] = 0.5 / SH_C0
grid.sh_data.data = grid.sh_data.data.flatten(1, 2)
grid.density_data.data[:] = 1.0

if grid.use_background:
	grid.background_data.data[..., -1] = 0.5
	grid.background_data.data[..., :-1] = torch.randn_like(
            grid.background_data.data[..., :-1]) * 0.01

if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE:
    grid.basis_data.data.normal_()
    grid.basis_data.data += 1.0

ENABLE_TORCH_CHECK = True
#  N_RAYS = 5000 #200 * 200
N_RAYS = 200 * 200
origins = torch.randn((N_RAYS, 3), device=device, dtype=dtype) * 3
dirs = torch.randn((N_RAYS, 3), device=device, dtype=dtype)

# from training.networks_plenoxel import RaySampler, PoseSampler
# img_res = 32
# ps = PoseSampler(range_azim=(180, 45), range_polar=(90, 15), radius=8, dist='normal')
# rs = RaySampler(img_resolution=img_res, fov=30)
# pose = ps.sample()
# origins, dirs = rs.sample_at(pose[None, :3, :4], rs.img_resolution)
# origins = origins.cuda()
# dirs = dirs.cuda()

#  origins = torch.clip(origins, -0.8, 0.8)

#  origins = torch.tensor([[-0.6747068762779236, -0.752697229385376, -0.800000011920929]], device=device, dtype=dtype)
#  dirs = torch.tensor([[0.6418760418891907, -0.37417781352996826, 0.6693176627159119]], device=device, dtype=dtype)
dirs /= torch.norm(dirs, dim=-1, keepdim=True)

#  start = 71
#  end = 72
#  origins = origins[start:end]
#  dirs = dirs[start:end]
#  print(origins.tolist(), dirs.tolist())

#  breakpoint()
rays = svox2.Rays(origins, dirs)

rgb_gt = torch.zeros((origins.size(0), 4), device=device, dtype=dtype)

#  grid.requires_grad_(True)

#  samps = grid.volume_render(rays, use_kernel=True)
#  sampt = grid.volume_render(grid, origins, dirs, use_kernel=False)

with Timing("ours"):
    samps = grid.volume_render(rays, use_kernel=True, render_alpha=True)
assert all([(a == samps[:, 0]).all() for a in samps.T[1:]]), 'Alpha map and rendered color images do not match.'

# Check internal gradients, i.e. white color vs alpha
density_grads_s = []
grid.sh_data.grad = None
grid.density_data.grad = None
for i in range(4):
    a = grid.volume_render(rays, use_kernel=True, render_alpha=True)[:, i]
    s = F.mse_loss(a, rgb_gt[:, i])
    with Timing("ours_backward"):
        s.backward()
    density_grads_s.append(grid.density_data.grad.clone().cpu())
    grid.sh_data.grad = None
    grid.density_data.grad = None
density_grads_s = torch.stack(density_grads_s)
print('Error gradients color/alpha', [(g - density_grads_s[0]).abs().max().item() for g in density_grads_s[1:]])

# Check gradients wrt torch implementation
s = F.mse_loss(samps, rgb_gt)

print(s)
print('bkwd..')
with Timing("ours_backward"):
    s.backward()
grid_sh_grad_s = grid.sh_data.grad.clone().cpu()
grid_density_grad_s = grid.density_data.grad.clone().cpu()
grid.sh_data.grad = None
grid.density_data.grad = None
if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE:
    grid_basis_grad_s = grid.basis_data.grad.clone().cpu()
    grid.basis_data.grad = None
if grid.use_background:
    grid_bg_grad_s = grid.background_data.grad.clone().cpu()
    grid.background_data.grad = None

if ENABLE_TORCH_CHECK:
    with Timing("torch"):
        sampt = grid.volume_render(rays, use_kernel=False, render_alpha=True)
    assert all([(a == sampt[:, 0]).all() for a in sampt.T[1:]]), 'Alpha map and rendered color images do not match.'

    print('Do ours and torch output match?', torch.isclose(samps, sampt).all().item())

    s = F.mse_loss(sampt, rgb_gt)
    with Timing("torch_backward"):
        s.backward()
    grid_sh_grad_t = grid.sh_data.grad.clone().cpu() if grid.sh_data.grad is not None else torch.zeros_like(grid_sh_grad_s)
    grid_density_grad_t = grid.density_data.grad.clone().cpu() if grid.density_data.grad is not None else torch.zeros_like(grid_density_grad_s)
    if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE:
        grid_basis_grad_t = grid.basis_data.grad.clone().cpu()
    if grid.use_background:
        grid_bg_grad_t = grid.background_data.grad.clone().cpu() if grid.background_data.grad is not None else torch.zeros_like(grid_bg_grad_s)

    E = torch.abs(grid_sh_grad_s-grid_sh_grad_t)
    Ed = torch.abs(grid_density_grad_s-grid_density_grad_t)
    if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE:
        Eb = torch.abs(grid_basis_grad_s-grid_basis_grad_t)
    if grid.use_background:
        Ebg = torch.abs(grid_bg_grad_s-grid_bg_grad_t)
    print('err', torch.abs(samps - sampt).max())
    print('err_sh_grad\n', E.max())
    print(' mean\n', E.mean())
    print('err_density_grad\n', Ed.max())
    print(' mean\n', Ed.mean())
    if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE:
        print('err_basis_grad\n', Eb.max())
        print(' mean\n', Eb.mean())
    if grid.use_background:
        print('err_background_grad\n', Ebg.max())
        print(' mean\n', Ebg.mean())
    print()
    print('g_ours sh min/max\n', grid_sh_grad_s.min(), grid_sh_grad_s.max())
    print('g_torch sh min/max\n', grid_sh_grad_t.min(), grid_sh_grad_t.max())
    print('g_ours sigma min/max\n', grid_density_grad_s.min(), grid_density_grad_s.max())
    print('g_torch sigma min/max\n', grid_density_grad_t.min(), grid_density_grad_t.max())
    if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE:
        print('g_ours basis min/max\n', grid_basis_grad_s.min(), grid_basis_grad_s.max())
        print('g_torch basis min/max\n', grid_basis_grad_t.min(), grid_basis_grad_t.max())
    if grid.use_background:
        print('g_ours bg min/max\n', grid_bg_grad_s.min(), grid_bg_grad_s.max())
        print('g_torch bg min/max\n', grid_bg_grad_t.min(), grid_bg_grad_t.max())
