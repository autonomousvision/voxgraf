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

"""Train a GAN using the techniques described in the paper
"Alias-Free Generative Adversarial Networks"."""

import os
import shutil
import click
import re
import json
import tempfile
import torch
import legacy
import dill
import warnings

# environment variables
os.environ['OMP_NUM_THREADS'] = "16"

import dnnlib
from training import training_loop
from metrics import metric_main
from torch_utils import training_stats
from torch_utils import custom_ops
from torch_utils import misc

from svox2 import svox2

#----------------------------------------------------------------------------

def subprocess_fn(rank, c, temp_dir):
    dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    if rank == 0:
        write_gpu_info()
        print(c.run_dir)

    # Init torch.distributed.
    if c.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=c.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=c.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if c.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    # Execute training loop.
    training_loop.training_loop(rank=rank, **c)

#----------------------------------------------------------------------------

def launch_training(c, desc, outdir, dry_run):
    dnnlib.util.Logger(should_flush=True)

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]

    matching_dirs = [re.fullmatch(r'\d{5}' + f'-{desc}', x) for x in prev_run_dirs if
                     re.fullmatch(r'\d{5}' + f'-{desc}', x) is not None]
    if c.continue_training and len(matching_dirs) > 0:  # expect unique desc, continue in this directory
        assert len(matching_dirs) == 1, f'Multiple directories found for resuming: {matching_dirs}'
        c.run_dir = os.path.join(outdir, matching_dirs[0].group())
    else:  # fallback to standard
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        c.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{desc}')
        assert not os.path.exists(c.run_dir)

    # Print options.
    print()
    print('Training options:')
    print(json.dumps(c, indent=2))
    print()
    print(f'Output directory:    {c.run_dir}')
    print(f'Number of GPUs:      {c.num_gpus}')
    print(f'Batch size:          {c.batch_size} images')
    print(f'Training duration:   {c.total_kimg} kimg')
    print(f'Dataset path:        {c.training_set_kwargs.path}')
    print(f'Dataset size:        {c.training_set_kwargs.max_size} images')
    print(f'Dataset resolution:  {c.training_set_kwargs.resolution}')
    print(f'Dataset labels:      {c.training_set_kwargs.use_labels}')
    print(f'Dataset x-flips:     {c.training_set_kwargs.xflip}')
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    print('Creating output directory...')
    os.makedirs(c.run_dir, exist_ok=c.continue_training)
    with open(os.path.join(c.run_dir, 'training_options.json'), 'wt+') as f:
        json.dump(c, f, indent=2)

    if c.resume_pkl is not None:            # create local copy and set best_fid to 9999 to ensure saving new best model
        resume_file = os.path.join(c.run_dir, 'model_start.pkl')
        if not os.path.isfile(resume_file):
            shutil.copyfile(c.resume_pkl, resume_file)
        else:
            'Local copy of resume file exists - start from there and ignore "resume_pkl" input. ' \
            'If this is not desired, please rename or remove the local copy first.'
        c.resume_pkl = resume_file
        with dnnlib.util.open_url(resume_file) as f:
            resume_data = legacy.load_network_pkl(f)
            resume_data['progress']['best_fid'] = 9999
        with open(resume_file, 'wb') as f:
            dill.dump(resume_data, f)

    # Launch processes.
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if c.num_gpus == 1:
            subprocess_fn(rank=0, c=c, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(c, temp_dir), nprocs=c.num_gpus)

#----------------------------------------------------------------------------

def init_dataset_kwargs(data):
    class_name = 'training.dataset.ImageFolderDataset'
    try:
        dataset_kwargs = dnnlib.EasyDict(class_name=class_name, path=data, use_labels=False, max_size=None, xflip=False)
        dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # Subclass of training.dataset.Dataset.
        if dataset_obj.label_dim != 0: warnings.warn('Ignore labels.')
        dataset_kwargs.resolution = dataset_obj.resolution # Be explicit about resolution.
        dataset_kwargs.max_size = len(dataset_obj) # Be explicit about dataset size.
        return dataset_kwargs, dataset_obj.name
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

#----------------------------------------------------------------------------

def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')

#----------------------------------------------------------------------------

def write_gpu_info():
    import subprocess as sp
    print(sp.getoutput('nvidia-smi'))

#----------------------------------------------------------------------------

@click.command()

# Required.
@click.option('--outdir',       help='Where to save the results', metavar='DIR',                required=True)
@click.option('--data',         help='Training data', metavar='[ZIP|DIR]',                      type=str, required=True)
@click.option('--gpus',         help='Number of GPUs to use', metavar='INT',                    type=click.IntRange(min=1), required=True)
@click.option('--batch',        help='Total batch size', metavar='INT',                         type=click.IntRange(min=1), required=True)
@click.option('--gamma',        help='R1 regularization weight', metavar='FLOAT',               type=click.FloatRange(min=0), default=1.0)

# Optional features.
@click.option('--aug',          help='Augmentation mode',                                       type=click.Choice(['noaug', 'ada', 'fixed']), default='ada', show_default=True)
@click.option('--resume',       help='Resume from given network pickle', metavar='[PATH|URL]',  type=str)
@click.option('--continue-training', help='If output directory with same name exists, continue training there.', type=bool, default=True, show_default=True)

# Voxgraf settings.
@click.option('--grid-res',     help='Resolution of sparse voxel grid', metavar='INT',          type=click.IntRange(min=1), default=32, show_default=True)
@click.option('--dense-resolution', help='resolution of dense stem for hybrid arch', metavar='INT', type=click.IntRange(min=4), default=32, show_default=True)
@click.option('--prune-last',   help='Prune the last layer',                                      is_flag=True)
@click.option('--sparse_type',  help='Implementation of sparse convolution',                    type=click.Choice(['pytorch', 'minkowski']), default='pytorch', show_default=True)
@click.option('--sigma_mpl',    help='Sigma multiplier',                                        type=click.FloatRange(), default=30, show_default=True)
@click.option('--n-refinement', help='Number of refinement layers',                             type=click.IntRange(min=0), default=3, show_default=True)
@click.option('--use_bg',       help='Generate a background',                                   type=bool, default=True, show_default=True)
@click.option('--bg_cbase',     help='Capacity multiplier', metavar='INT',                      type=click.IntRange(min=1), default=2048, show_default=True)
@click.option('--bg_cmax',      help='Max. feature maps', metavar='INT',                        type=click.IntRange(min=1), default=64, show_default=True)

# Regularizers
@click.option('--lambda_tv_sigma',  help='Strength of total variation loss on density', metavar='FLOAT',    type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--lambda_cvg_fg',    help='Strength of foreground coverage loss', metavar='FLOAT',           type=click.FloatRange(min=0), default=0.1, show_default=True)
@click.option('--min_cvg_fg',       help='Minimum foreground coverage threshold', metavar='FLOAT',          type=click.FloatRange(min=0), default=0.4, show_default=True)
@click.option('--lambda_cvg_bg',    help='Strength of background coverage loss', metavar='FLOAT',           type=click.FloatRange(min=0), default=0.1, show_default=True)
@click.option('--min_cvg_bg',       help='Minimum background coverage threshold', metavar='FLOAT',          type=click.FloatRange(min=0), default=0.1, show_default=True)
@click.option('--lambda_vardepth',  help='Strength of depth variance loss', metavar='FLOAT',                 type=click.FloatRange(min=0), default=1e-2, show_default=True)
@click.option('--lambda_sparsity',  help='Strength of sparsity regularization', metavar='FLOAT',            type=click.FloatRange(min=0), default=0, show_default=True)

# Misc hyperparameters.
@click.option('--p',            help='Probability for --aug=fixed', metavar='FLOAT',            type=click.FloatRange(min=0, max=1), default=0.2, show_default=True)
@click.option('--target',       help='Target value for --aug=ada', metavar='FLOAT',             type=click.FloatRange(min=0, max=1), default=0.6, show_default=True)
@click.option('--batch-gpu',    help='Limit batch size per GPU', metavar='INT',                 type=click.IntRange(min=1))
@click.option('--cbase_d',      help='Capacity multiplier (D)', metavar='INT',                  type=click.IntRange(min=1), default=32768, show_default=True)
@click.option('--cmax_d',       help='Max. feature maps', metavar='INT',                        type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--cbase_g',      help='Capacity multiplier (G)', metavar='INT',                  type=click.IntRange(min=1), default=4000 , show_default=True)
@click.option('--cmax_g',       help='Max. feature maps', metavar='INT',                        type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--glr',          help='G learning rate  [default: varies]', metavar='FLOAT',     type=click.FloatRange(min=0), default=0.0025, show_default=True)
@click.option('--dlr',          help='D learning rate', metavar='FLOAT',                        type=click.FloatRange(min=0), default=0.002, show_default=True)
@click.option('--mbstd-group',  help='Minibatch std group size', metavar='INT',                 type=click.IntRange(min=1), default=4, show_default=True)

# Misc settings.
@click.option('--desc',         help='String to include in result dir name', metavar='STR',     type=str)
@click.option('--metrics',      help='Quality metrics', metavar='[NAME|A,B,C|none]',            type=parse_comma_separated_list, default='fid10k_full', show_default=True)
@click.option('--kimg',         help='Total training duration', metavar='KIMG',                 type=click.IntRange(min=1), default=25000, show_default=True)
@click.option('--tick',         help='How often to print progress', metavar='KIMG',             type=click.FloatRange(min=0.001), default=4, show_default=True)
@click.option('--snap',         help='How often to save snapshots', metavar='TICKS',            type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--snap_ckpt',    help='How often to save checkpoints', metavar='TICKS',          type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--seed',         help='Random seed', metavar='INT',                              type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--fp32',         help='Disable mixed-precision', metavar='BOOL',                 type=bool, default=False, show_default=True)
@click.option('--nobench',      help='Disable cuDNN benchmarking', metavar='BOOL',              type=bool, default=False, show_default=True)
@click.option('--workers',      help='DataLoader worker processes', metavar='INT',              type=click.IntRange(min=1), default=3, show_default=True)
@click.option('-n','--dry-run', help='Print training options and exit',                         is_flag=True)

def main(**kwargs):
    """Train a GAN using the techniques described in the paper
    "Alias-Free Generative Adversarial Networks".

    Examples:

    \b
    # Train StyleGAN3-T for AFHQv2 using 8 GPUs.
    python train.py --outdir=~/training-runs --cfg=stylegan3-t --data=~/datasets/afhqv2-512x512.zip \\
        --gpus=8 --batch=32 --gamma=8.2 --mirror=1

    \b
    # Fine-tune StyleGAN3-R for MetFaces-U using 1 GPU, starting from the pre-trained FFHQ-U pickle.
    python train.py --outdir=~/training-runs --cfg=stylegan3-r --data=~/datasets/metfacesu-1024x1024.zip \\
        --gpus=8 --batch=32 --gamma=6.6 --mirror=1 --kimg=5000 --snap=5 \\
        --resume=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhqu-1024x1024.pkl

    \b
    # Train StyleGAN2 for FFHQ at 1024x1024 resolution using 8 GPUs.
    python train.py --outdir=~/training-runs --cfg=stylegan2 --data=~/datasets/ffhq-1024x1024.zip \\
        --gpus=8 --batch=32 --gamma=10 --mirror=1 --aug=noaug
    """

    # Initialize config.
    opts = dnnlib.EasyDict(kwargs) # Command line arguments.
    c = dnnlib.EasyDict() # Main config dict.
    c.G_kwargs = dnnlib.EasyDict(class_name='training.networks_voxgraf.Generator', z_dim=64, w_dim=64, mapping_kwargs=dnnlib.EasyDict())
    c.D_kwargs = dnnlib.EasyDict(class_name='training.networks_stylegan2.Discriminator', block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict(), architecture='skip')
    c.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8)
    c.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8)
    c.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.VoxGRAFLoss')
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, prefetch_factor=2)

    # Training set.
    c.training_set_kwargs, dataset_name = init_dataset_kwargs(data=opts.data)

    # Camera settings
    if dataset_name.startswith('carla'):
        c.G_kwargs.pose_kwargs = dnnlib.EasyDict(range_azim=(0, 360), range_polar=(0, 90), dist='uniform', radius=10.0)
        dataset_fov = 30.0
        grid_radius = [2.5, 2.5, 2.5]
    elif dataset_name.startswith('ffhq') or dataset_name.startswith('afhq') or dataset_name.startswith('grafcats'):
        c.G_kwargs.pose_kwargs = dnnlib.EasyDict(range_azim=(180, 20), range_polar=(90, 10), dist='normal', radius=10.0)
        dataset_fov = 12.6      # from face estimator
        grid_radius = [1.2, 1.2, 1.2]  # estimate grid size via sin(fov/2)*r * sqrt(2) -> 1.6
    else:
        raise AttributeError(f'Please specify camera parameters for dataset {dataset_name}.')
    c.G_kwargs.render_kwargs = dnnlib.EasyDict(
        grid_radius=grid_radius, grid_center=[0, 0, 0], fov=dataset_fov, basis_type=svox2.BASIS_TYPE_SH, basis_dim=1,
        opt_kwargs=dnnlib.EasyDict(backend='cuvol', background_brightness=1, step_size=0.5, sigma_thresh=1e-08, stop_thresh=1e-7)
    )

    # Hyperparameters & settings.
    c.num_gpus = opts.gpus
    c.batch_size = opts.batch
    c.batch_gpu = opts.batch_gpu or opts.batch // opts.gpus
    c.D_kwargs.channel_base = opts.cbase_d
    c.D_kwargs.channel_max = opts.cmax_d

    c.G_kwargs.mapping_kwargs.num_layers = 2

    c.G_kwargs.use_bg = opts.use_bg
    c.G_kwargs.sigma_mpl = opts.sigma_mpl

    c.D_kwargs.block_kwargs.freeze_layers = 0
    c.D_kwargs.epilogue_kwargs.mbstd_group_size = opts.mbstd_group
    c.loss_kwargs.r1_gamma = opts.gamma
    c.G_kwargs.pose_conditioning = c.loss_kwargs.generator_pose_conditioning = True
    c.loss_kwargs.p_generator_pose_conditioning = 0.5
    c.G_kwargs.c_dim = 12 if c.G_kwargs.pose_conditioning else 0
    c.loss_kwargs.discriminator_pose_conditioning = True
    c.D_kwargs.c_dim = 12 if c.loss_kwargs.discriminator_pose_conditioning else 0
    c.D_kwargs.mapping_kwargs.num_layers = 2
    c.D_reg_interval = 4
    c.G_reg_interval = None         # G regularization is executed with Gmain
    c.G_opt_kwargs.lr = opts.glr
    c.D_opt_kwargs.lr = opts.dlr
    c.metrics = opts.metrics
    c.total_kimg = opts.kimg
    c.kimg_per_tick = opts.tick
    c.image_snapshot_ticks = c.network_snapshot_ticks = opts.snap
    c.network_ckpt_ticks = None if opts.snap_ckpt == 0 else opts.snap_ckpt
    c.random_seed = c.training_set_kwargs.random_seed = opts.seed
    c.data_loader_kwargs.num_workers = opts.workers

    # Sanity checks.
    if c.batch_size % c.num_gpus != 0:
        raise click.ClickException('--batch must be a multiple of --gpus')
    if c.batch_size % (c.num_gpus * c.batch_gpu) != 0:
        raise click.ClickException('--batch must be a multiple of --gpus times --batch-gpu')
    if c.batch_gpu < c.D_kwargs.epilogue_kwargs.mbstd_group_size:
        raise click.ClickException('--batch-gpu cannot be smaller than --mbstd')
    if any(not metric_main.is_valid_metric(metric) for metric in c.metrics):
        raise click.ClickException('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))

    c.ema_kimg = c.batch_size * 10 / 32

    # VoxGRAF settings
    c.G_kwargs.fg_kwargs = dnnlib.EasyDict(channel_base=opts.cbase_g, channel_max=opts.cmax_g, grid_resolution=opts.grid_res, dense_resolution=opts.dense_resolution, prune_last=opts.prune_last, sparse_type=opts.sparse_type)
    if c.G_kwargs.fg_kwargs.sparse_type == 'minkowski':
        try:
            import MinkowskiEngine
        except:
            raise ImportError('Please install Minkowski Engine when using sparse_type=="minkowski"')
    c.G_kwargs.bg_kwargs = dnnlib.EasyDict(channel_base=opts.bg_cbase, channel_max=opts.bg_cmax, fused_modconv_default='inference_only')
    c.G_kwargs.refinement_kwargs = dnnlib.EasyDict(num_layers=opts.n_refinement, dhidden=16)

    # Regularizers
    c.loss_kwargs.raw_noise_std = 1.                    # add noise to sigma values in the beginning of training
    c.loss_kwargs.decrease_noise_until = 5000
    c.loss_kwargs.no_reg_until = 5000
    c.loss_kwargs.tv_kwargs = dnnlib.EasyDict(lambda_tv=opts.lambda_tv_sigma)
    c.loss_kwargs.cvg_kwargs = dnnlib.EasyDict(lambda_cvg_fg=opts.lambda_cvg_fg, min_cvg_fg=opts.min_cvg_fg, lambda_cvg_bg=opts.lambda_cvg_bg, min_cvg_bg=opts.min_cvg_bg)
    c.loss_kwargs.lambda_vardepth = opts.lambda_vardepth
    c.loss_kwargs.lambda_sparsity = opts.lambda_sparsity

    # Augmentation.
    if opts.aug != 'noaug':
        c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', xflip=0, rotate90=0, xint=1, scale=1, rotate=0, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1)
        if opts.aug == 'ada':
            c.ada_target = opts.target
        if opts.aug == 'fixed':
            c.augment_p = opts.p

    # Resume.
    if opts.resume is not None:
        c.resume_pkl = opts.resume
        c.ada_kimg = 100 # Make ADA react faster at the beginning.
        c.ema_rampup = None # Disable EMA rampup.
    else:
        c.resume_pkl = None

    # Continue.
    c.continue_training = opts.continue_training

    # Performance-related toggles.
    if opts.fp32:
        c.G_kwargs.num_fp16_res = c.G_kwargs.bg_kwargs.num_fp16_res = c.D_kwargs.num_fp16_res = 0
        c.G_kwargs.conv_clamp = c.G_kwargs.bg_kwargs.conv_clamp = c.D_kwargs.conv_clamp = None
    if opts.nobench:
        c.cudnn_benchmark = False

    # Description string.
    desc = f'{dataset_name:s}-grid{c.G_kwargs.fg_kwargs.grid_resolution}-batch{c.batch_size:d}-gamma{c.loss_kwargs.r1_gamma:g}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    # Launch.
    launch_training(c=c, desc=desc, outdir=opts.outdir, dry_run=opts.dry_run)

    # Check for restart
    last_snapshot = misc.get_ckpt_path(c.run_dir)
    if os.path.isfile(last_snapshot):
        # get current number of training images
        with dnnlib.util.open_url(last_snapshot) as f:
            cur_nimg = legacy.load_network_pkl(f)['progress']['cur_nimg'].item()
        if (cur_nimg//1000) < c.total_kimg:
            print('Restart: exit with code 3')
            exit(3)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
