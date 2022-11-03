# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import re
import contextlib
import numpy as np
import torch
import warnings
import dnnlib

#----------------------------------------------------------------------------
# Cached construction of constant tensors. Avoids CPU=>GPU copy when the
# same constant is used multiple times.

_constant_cache = dict()

def constant(value, shape=None, dtype=None, device=None, memory_format=None):
    value = np.asarray(value)
    if shape is not None:
        shape = tuple(shape)
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device('cpu')
    if memory_format is None:
        memory_format = torch.contiguous_format

    key = (value.shape, value.dtype, value.tobytes(), shape, dtype, device, memory_format)
    tensor = _constant_cache.get(key, None)
    if tensor is None:
        tensor = torch.as_tensor(value.copy(), dtype=dtype, device=device)
        if shape is not None:
            tensor, _ = torch.broadcast_tensors(tensor, torch.empty(shape))
        tensor = tensor.contiguous(memory_format=memory_format)
        _constant_cache[key] = tensor
    return tensor

#----------------------------------------------------------------------------
# Replace NaN/Inf with specified numerical values.

try:
    nan_to_num = torch.nan_to_num # 1.8.0a0
except AttributeError:
    def nan_to_num(input, nan=0.0, posinf=None, neginf=None, *, out=None): # pylint: disable=redefined-builtin
        assert isinstance(input, torch.Tensor)
        if posinf is None:
            posinf = torch.finfo(input.dtype).max
        if neginf is None:
            neginf = torch.finfo(input.dtype).min
        assert nan == 0
        return torch.clamp(input.unsqueeze(0).nansum(0), min=neginf, max=posinf, out=out)

#----------------------------------------------------------------------------
# Symbolic assert.

try:
    symbolic_assert = torch._assert # 1.8.0a0 # pylint: disable=protected-access
except AttributeError:
    symbolic_assert = torch.Assert # 1.7.0

#----------------------------------------------------------------------------
# Context manager to temporarily suppress known warnings in torch.jit.trace().
# Note: Cannot use catch_warnings because of https://bugs.python.org/issue29672

@contextlib.contextmanager
def suppress_tracer_warnings():
    flt = ('ignore', None, torch.jit.TracerWarning, None, 0)
    warnings.filters.insert(0, flt)
    yield
    warnings.filters.remove(flt)

#----------------------------------------------------------------------------
# Assert that the shape of a tensor matches the given list of integers.
# None indicates that the size of a dimension is allowed to vary.
# Performs symbolic assertion when used in torch.jit.trace().

def assert_shape(tensor, ref_shape):
    if tensor.ndim != len(ref_shape):
        raise AssertionError(f'Wrong number of dimensions: got {tensor.ndim}, expected {len(ref_shape)}')
    for idx, (size, ref_size) in enumerate(zip(tensor.shape, ref_shape)):
        if ref_size is None:
            pass
        elif isinstance(ref_size, torch.Tensor):
            with suppress_tracer_warnings(): # as_tensor results are registered as constants
                symbolic_assert(torch.equal(torch.as_tensor(size), ref_size), f'Wrong size for dimension {idx}')
        elif isinstance(size, torch.Tensor):
            with suppress_tracer_warnings(): # as_tensor results are registered as constants
                symbolic_assert(torch.equal(size, torch.as_tensor(ref_size)), f'Wrong size for dimension {idx}: expected {ref_size}')
        elif size != ref_size:
            raise AssertionError(f'Wrong size for dimension {idx}: got {size}, expected {ref_size}')

#----------------------------------------------------------------------------
# Function decorator that calls torch.autograd.profiler.record_function().

def profiled_function(fn):
    def decorator(*args, **kwargs):
        with torch.autograd.profiler.record_function(fn.__name__):
            return fn(*args, **kwargs)
    decorator.__name__ = fn.__name__
    return decorator

#----------------------------------------------------------------------------
# Sampler for torch.utils.data.DataLoader that loops over the dataset
# indefinitely, shuffling items as it goes.
import random

class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, rank=0, num_replicas=1, shuffle=True, seed=0, window_size=0.5):
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        assert 0 <= window_size <= 1
        super().__init__(dataset)
        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size

    def __iter__(self):
        order = np.arange(len(self.dataset))
        rnd = None
        window = 0
        if self.shuffle:
            rnd = np.random.RandomState(self.seed)
            rnd.shuffle(order)
            window = int(np.rint(order.size * self.window_size))

        idx = 0
        while True:
            i = idx % order.size
            if idx % self.num_replicas == self.rank:
                yield order[i]
            if window >= 2:
                j = (i - rnd.randint(window)) % order.size
                order[i], order[j] = order[j], order[i]
            idx += 1

#----------------------------------------------------------------------------
# Utilities for operating with torch.nn.Module parameters and buffers.

def params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.parameters()) + list(module.buffers())

def named_params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.named_parameters()) + list(module.named_buffers())

def copy_params_and_buffers(src_module, dst_module, require_all=False):
    assert isinstance(src_module, torch.nn.Module)
    assert isinstance(dst_module, torch.nn.Module)
    src_tensors = dict(named_params_and_buffers(src_module))
    for name, tensor in named_params_and_buffers(dst_module):
        assert (name in src_tensors) or (not require_all)
        if name in src_tensors:
            try:
                tensor.copy_(src_tensors[name].detach()).requires_grad_(tensor.requires_grad)
            except:
                if not name in [n for n, _ in dst_module.named_buffers()]:        # fine for mask buffers
                    raise
                warnings.warn(f'Do not copy buffer {name}')

#----------------------------------------------------------------------------
# Context manager for easily enabling/disabling DistributedDataParallel
# synchronization.

@contextlib.contextmanager
def ddp_sync(module, sync):
    assert isinstance(module, torch.nn.Module)
    if sync or not isinstance(module, torch.nn.parallel.DistributedDataParallel):
        yield
    else:
        with module.no_sync():
            yield

#----------------------------------------------------------------------------
# Check DistributedDataParallel consistency across processes.

def check_ddp_consistency(module, ignore_regex=None):
    assert isinstance(module, torch.nn.Module)
    for name, tensor in named_params_and_buffers(module):
        fullname = type(module).__name__ + '.' + name
        if ignore_regex is not None and re.fullmatch(ignore_regex, fullname):
            continue
        tensor = tensor.detach()
        if tensor.is_floating_point():
            tensor = nan_to_num(tensor)
        other = tensor.clone()
        torch.distributed.broadcast(tensor=other, src=0)
        assert (tensor == other).all(), fullname

#----------------------------------------------------------------------------
# Print summary table of module hierarchy.

def print_module_summary(module, inputs, max_nesting=3, skip_redundant=True):
    assert isinstance(module, torch.nn.Module)
    assert not isinstance(module, torch.jit.ScriptModule)
    assert isinstance(inputs, (tuple, list))

    # Register hooks.
    entries = []
    nesting = [0]
    def pre_hook(_mod, _inputs):
        nesting[0] += 1
    def post_hook(mod, _inputs, outputs):
        nesting[0] -= 1
        if nesting[0] <= max_nesting:
            outputs = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
            outputs = [t for t in outputs if isinstance(t, torch.Tensor)]
            entries.append(dnnlib.EasyDict(mod=mod, outputs=outputs))
    hooks = [mod.register_forward_pre_hook(pre_hook) for mod in module.modules()]
    hooks += [mod.register_forward_hook(post_hook) for mod in module.modules()]

    # Run module.
    outputs = module(*inputs)
    for hook in hooks:
        hook.remove()

    # Identify unique outputs, parameters, and buffers.
    tensors_seen = set()
    for e in entries:
        e.unique_params = [t for t in e.mod.parameters() if id(t) not in tensors_seen]
        e.unique_buffers = [t for t in e.mod.buffers() if id(t) not in tensors_seen]
        e.unique_outputs = [t for t in e.outputs if id(t) not in tensors_seen]
        tensors_seen |= {id(t) for t in e.unique_params + e.unique_buffers + e.unique_outputs}

    # Filter out redundant entries.
    if skip_redundant:
        entries = [e for e in entries if len(e.unique_params) or len(e.unique_buffers) or len(e.unique_outputs)]

    # Construct table.
    rows = [[type(module).__name__, 'Parameters', 'Buffers', 'Output shape', 'Datatype']]
    rows += [['---'] * len(rows[0])]
    param_total = 0
    buffer_total = 0
    submodule_names = {mod: name for name, mod in module.named_modules()}
    for e in entries:
        name = '<top-level>' if e.mod is module else submodule_names[e.mod]
        param_size = sum(t.numel() for t in e.unique_params)
        buffer_size = sum(t.numel() for t in e.unique_buffers)
        output_shapes = [str(list(t.shape)) for t in e.outputs]
        output_dtypes = [str(t.dtype).split('.')[-1] for t in e.outputs]
        rows += [[
            name + (':0' if len(e.outputs) >= 2 else ''),
            str(param_size) if param_size else '-',
            str(buffer_size) if buffer_size else '-',
            (output_shapes + ['-'])[0],
            (output_dtypes + ['-'])[0],
        ]]
        for idx in range(1, len(e.outputs)):
            rows += [[name + f':{idx}', '-', '-', output_shapes[idx], output_dtypes[idx]]]
        param_total += param_size
        buffer_total += buffer_size
    rows += [['---'] * len(rows[0])]
    rows += [['Total', str(param_total), str(buffer_total), '-', '-']]

    # Print table.
    widths = [max(len(cell) for cell in column) for column in zip(*rows)]
    print()
    for row in rows:
        print('  '.join(cell + ' ' * (width - len(cell)) for cell, width in zip(row, widths)))
    print()
    return outputs

#----------------------------------------------------------------------------

# Added by Katja
import os
import gc
from tqdm import tqdm

def get_ckpt_path(run_dir):
    return os.path.join(run_dir, f'network-snapshot.pkl')

# for progressive growing
def assert_sorted_dict(dict):
    assert list(dict.keys()) == sorted(dict.keys())
    assert list(dict.values()) == sorted(dict.values())

def nimg_to_lod(cur_nimg, schedule):
    steps = list(schedule.keys())
    assert min(steps) == 0
    assert sorted(steps) == steps

    cur_step = max([s if s <= cur_nimg else 0 for s in steps])

    lod = steps.index(cur_step)
    res = schedule[cur_step]

    if cur_step == max(steps):
        weight = 1
    else:
        next_step = steps[lod+1]
        weight = (cur_nimg - cur_step) / (next_step - cur_step)

    return lod, res, weight


def check_for_resolution_change(cur_nimg, schedule, batch_size):
    _, cur_res, _ = nimg_to_lod(cur_nimg, schedule)
    _, next_res, _ = nimg_to_lod(cur_nimg+batch_size, schedule)
    return cur_res != next_res


def plot_svg(svg, resolution):
    import matplotlib.pyplot as plt;
    plt.ion()
    import numpy as np
    fig = plt.figure()
    ax_3d = fig.add_subplot(1, 1, 1, projection='3d')
    ax_3d.grid(False)
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')

    batch_coords, batch_feats = svg.decomposed_coordinates_and_features
    coords, feats = batch_coords[0].detach().cpu(), batch_feats[0].detach().cpu()

    vox = np.zeros((resolution, resolution, resolution), dtype=bool)
    coords = (coords / torch.tensor(svg.tensor_stride).view(1, 3)).to(torch.long)
    if feats.shape[1] == 1:     # density
        mask = feats > 0
        coords = coords[mask.expand(-1, 3)].view(-1, 3)
        feats_val = feats[mask].view(-1, 1)
        feats = torch.zeros_like(feats_val.expand(-1, 3))
        feats[:, 2:3] = feats_val           # fill features with blue color

    vox[coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    colors = torch.zeros((*vox.shape, 4))
    colors[coords[:, 0], coords[:, 1], coords[:, 2]] = torch.cat([feats[:, :3].clip(0, 1), 0.3*torch.ones((feats.shape[0], 1))], axis=1)

    ax_3d.voxels(vox, facecolors=colors.numpy(), edgecolor='k')
    return ax_3d


class Memory:
    """
    Memory environment
    usage:
    with Memory("message"):
        your commands here
    will print peak CUDA memory in Mb
    """

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        torch.cuda.reset_max_memory_allocated()
        gc.collect()
        torch.cuda.synchronize()
        self.cur_mem = torch.cuda.max_memory_allocated()
        self.start = torch.cuda.Event()
        self.end = torch.cuda.Event()
        self.start.record()

    def __exit__(self, type, value, traceback):
        self.end.record()
        gc.collect()
        torch.cuda.synchronize()
        print(self.name, "occupied", (torch.cuda.max_memory_allocated() - self.cur_mem) / 1048576, "Mb")


def export_svg(coordinates, outfile, face_size=1):
    # from https://stackoverflow.com/questions/70660115/how-to-visualize-voxels-and-show-it-on-meshlab
    from plyfile import PlyData, PlyElement

    def write_ply(points, face_data, filename, text=True):
        points = [(points[i, 0], points[i, 1], points[i, 2]) for i in range(points.shape[0])]

        vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

        face = np.empty(len(face_data), dtype=[('vertex_indices', 'i4', (4,))])
        face['vertex_indices'] = face_data

        ply_faces = PlyElement.describe(face, 'face')
        ply_vertexs = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
        print('Save grid...')
        PlyData([ply_vertexs, ply_faces], text=text).write(filename)
        print('Done.')

    def occ2points(coordinates):
        points = []
        len = coordinates.shape[0]
        for i in range(len):
            points.append(np.array([int(coordinates[i, 1]), int(coordinates[i, 2]), int(coordinates[i, 3])]))

        return np.array(points)

    def generate_faces(points, face_size=1):
        corners = np.zeros((8 * len(points), 3))
        faces = np.zeros((6 * len(points), 4))
        dfhalf = 0.5*face_size
        for index in tqdm(range(len(points)), desc='Generate faces...'):
            corners[index * 8] = np.array([points[index, 0] - dfhalf, points[index, 1] - dfhalf, points[index, 2] - dfhalf])
            corners[index * 8 + 1] = np.array([points[index, 0] + dfhalf, points[index, 1] - dfhalf, points[index, 2] - dfhalf])
            corners[index * 8 + 2] = np.array([points[index, 0] - dfhalf, points[index, 1] + dfhalf, points[index, 2] - dfhalf])
            corners[index * 8 + 3] = np.array([points[index, 0] + dfhalf, points[index, 1] + dfhalf, points[index, 2] - dfhalf])
            corners[index * 8 + 4] = np.array([points[index, 0] - dfhalf, points[index, 1] - dfhalf, points[index, 2] + dfhalf])
            corners[index * 8 + 5] = np.array([points[index, 0] + dfhalf, points[index, 1] - dfhalf, points[index, 2] + dfhalf])
            corners[index * 8 + 6] = np.array([points[index, 0] - dfhalf, points[index, 1] + dfhalf, points[index, 2] + dfhalf])
            corners[index * 8 + 7] = np.array([points[index, 0] + dfhalf, points[index, 1] + dfhalf, points[index, 2] + dfhalf])
            base=len(points)+8*index
            faces[index*6]= np.array([base+2, base+3,base+1,base+0])
            faces[index*6+1]= np.array([base+4, base+5, base+7,base+6])
            faces[index*6+2]= np.array([base+3, base+2, base+6,base+7])
            faces[index*6+3]= np.array([base+0, base+1, base+5,base+4])
            faces[index*6+4]= np.array([base+2, base+0,base+4,base+6])
            faces[index*6+5]= np.array([base+1, base+3,base+7,base+5])

        return corners, faces

    def writeocc(coordinates, outfile, face_size):
        points = occ2points(coordinates)
        # print(points.shape)
        corners, faces = generate_faces(points, face_size)
        if points.shape[0] == 0:
            print('the predicted mesh has zero point!')
        else:
            points = np.concatenate((points, corners), axis=0)
            write_ply(points, faces, outfile)

    writeocc(coordinates, outfile, face_size)


def mirror_pose(c2w, zero_angle=0):
    # apply extrinsic rotation to camera pose
    R, T = c2w[:3, :3], c2w[:3, 3]

    # get the rotation angle
    phi = np.arctan2(T[1], T[0]) * 180 / np.pi
    dphi = (zero_angle - phi) * 2
    dphi = dphi * np.pi / 180

    rotz = np.array([
        [np.cos(dphi), -np.sin(dphi), 0, 0],
        [np.sin(dphi), np.cos(dphi), 0, 0],
        [0, 0, 1., 0],
        [0, 0, 0, 1.]
    ])

    return rotz @ c2w
