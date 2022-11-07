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

"""Network architectures from the paper
"Analyzing and Improving the Image Quality of StyleGAN".
Matches the original implementation of configs E-F by Karras et al. at
https://github.com/NVlabs/stylegan2/blob/master/training/networks_stylegan2.py"""

import numpy as np
import torch
from torch_utils import misc
from torch_utils import persistence
from torch_utils.ops import upfirdn2d
from torch_utils.ops import bias_act
from torch_utils.ops import fma
import torch.nn.functional as F
import dnnlib
import warnings
from training.networks_stylegan2 import FullyConnectedLayer

try:
    import MinkowskiEngine as ME
    from MinkowskiEngineBackend._C import ConvolutionMode, PoolingMode
    from MinkowskiEngine.MinkowskiKernelGenerator import KernelGenerator
    from MinkowskiEngine.MinkowskiSparseTensor import _get_coordinate_map_key
except ImportError:
    # load dummy ME so that class definitions do not break
    ME = dnnlib.EasyDict(MinkowskiPruning=torch.nn.Module)
    warnings.warn('Minkowski Engine installation not found - this is fine as long as you do not use sparse_type="minkowski"')


#----------------------------------------------------------------------------
# Dense utils


__DENSE_RESOLUTION__ = None
__SPARSE_RESOLUTION__ = None


def denseconv3d(x, w, f=None, up=1, padding=0, groups=1, **kwargs):
    if up == 2:
        w = w.permute(1, 0, 2, 3, 4)
        x = F.conv_transpose3d(x, w, stride=2, padding=0, groups=groups)
        return x[..., :-1, :-1, :-1]
    elif up == 1:
        return F.conv3d(x, w, stride=1, padding=padding, groups=groups)
    else:
        raise NotImplementedError


@misc.profiled_function
def modulated_denseconv3d(
    x,                          # Input tensor of shape [batch_size, in_channels, in_height, in_width].
    weight,                     # Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
    styles,                     # Modulation coefficients of shape [batch_size, in_channels].
    noise           = None,     # Optional noise tensor to add to the output activations.
    up              = 1,        # Integer upsampling factor.
    down            = 1,        # Integer downsampling factor.
    padding         = 0,        # Padding with respect to the upsampled image.
    resample_filter = None,     # Low-pass filter to apply when resampling activations. Must be prepared beforehand by calling upfirdn2d.setup_filter().
    demodulate      = True,     # Apply weight demodulation?
    flip_weight     = True,     # False = convolution, True = correlation (matches torch.nn.functional.conv2d).
    fused_modconv   = True,     # Perform modulation, convolution, and demodulation as a single fused operation?
):
    batch_size = x.shape[0]
    out_channels, in_channels, kh, kw, kd = weight.shape
    misc.assert_shape(weight, [out_channels, in_channels, kh, kw, kd]) # [OIkk]
    misc.assert_shape(x, [batch_size, in_channels, None, None, None]) # [NIHW]
    misc.assert_shape(styles, [batch_size, in_channels]) # [NI]

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (1 / np.sqrt(in_channels * kh * kw * kd) / weight.norm(float('inf'), dim=[1,2,3,4], keepdim=True)) # max_Ikk
        styles = styles / styles.norm(float('inf'), dim=1, keepdim=True) # max_I

    # Calculate per-sample weights and demodulation coefficients.
    w = None
    dcoefs = None
    if demodulate or fused_modconv:
        w = weight.unsqueeze(0) # [NOIkkk]
        w = w * styles.reshape(batch_size, 1, -1, 1, 1, 1) # [NOIkkk]
    if demodulate:
        dcoefs = (w.square().sum(dim=[2,3,4,5]) + 1e-8).rsqrt() # [NO]
    if demodulate and fused_modconv:
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1, 1) # [NOIkkk]

    # Execute by scaling the activations before and after the convolution.
    if not fused_modconv:
        x = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1, 1)
        x = denseconv3d(x=x, w=weight.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, flip_weight=flip_weight)
        if demodulate and noise is not None:
            x = fma.fma(x, dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1, 1), noise.to(x.dtype))
        elif demodulate:
            x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1, 1)
        elif noise is not None:
            x = x.add_(noise.to(x.dtype))
        return x
    else:
        raise NotImplementedError


def dense_upsample3d(x, f=None, up=2, padding=0, flip_filter=False, gain=1, upsampling_mode='trilinear', impl='cuda'):
    r"""Upsample a batch of 3D grids.

    Args:
        x:           Float32/float64/float16 input tensor of the shape
                     `[batch_size, num_channels, in_height, in_width, in_depth]`.
        f:           unused.
        up:          Integer upsampling factor. Can be a single int or a list/tuple
                     `[x, y, z]` (default: 1).
        padding:     unused.
        flip_filter: unused.
        gain:        Overall scaling factor for signal magnitude (default: 1).
        impl:        unused.

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.
    """
    assert isinstance(up, int)
    assert padding == 0

    kwargs = {}
    if upsampling_mode != 'nearest':
        kwargs['align_corners'] = True
    return gain * torch.nn.functional.interpolate(x, scale_factor=up, mode=upsampling_mode, **kwargs)

#----------------------------------------------------------------------------
# Sparse utils

sparse_activation_funcs = {
    'linear':   dnnlib.EasyDict(func=lambda x, **_:         x,                                          def_alpha=0,    def_gain=1,             cuda_idx=1, ref='',  has_2nd_grad=False),
    'lrelu':    dnnlib.EasyDict(func=lambda x, alpha, **_:  ME.MinkowskiFunctional.leaky_relu(x, alpha),   def_alpha=0.2,  def_gain=np.sqrt(2),    cuda_idx=3, ref='y', has_2nd_grad=False),
}

def get_cur_min_max_coordinate(t, final_resolution, start_resolution):
    assert ((np.log2(final_resolution) % 1) == 0) and ((np.log2(start_resolution) % 1) == 0), f'resolutions {start_resolution}, {final_resolution} must be power of 2'
    edge_coordinates = 2 ** (torch.arange(np.log2(final_resolution)- np.log2(start_resolution))).int().flip(0)

    # calc idx of current layer
    map_resolution = t.coordinate_map_key.get_key()[0][0]
    idx_layer = torch.where(edge_coordinates == map_resolution)[0]

    # calc the current minimum coordinate and get according max coordinate
    if len(idx_layer) == 0:  # no transposed convolution happened yet, so min_coordinate is still 0
        curr_min_coord = 0
    else:
        curr_min_coord = sum(edge_coordinates[:idx_layer + 1])
    max_coord = final_resolution - curr_min_coord
    return curr_min_coord, max_coord

def to_dense(t, res, final_resolution=None, start_resolution=None):
    if start_resolution is None:
        global __DENSE_RESOLUTION__
        if __DENSE_RESOLUTION__ is None:
            raise RuntimeError('Please provide start_resolution or set global variable __DENSE_RESOLUTION__ before using this function')
        start_resolution = __DENSE_RESOLUTION__
    if final_resolution is None:
        global __SPARSE_RESOLUTION__
        if __SPARSE_RESOLUTION__ is None:
            raise RuntimeError('Please provide final_resolution or set global variable __SPARSE_RESOLUTION__ before using this function')
        final_resolution = __SPARSE_RESOLUTION__

    # find shift resulting from previous transposed convolutions
    min_coord, _ = get_cur_min_max_coordinate(t, final_resolution, start_resolution)

    coordinates = t.C.clone()
    coordinates[:, 1:] += min_coord
    t_pos = ME.SparseTensor(
        features=t.F,
        coordinates=coordinates,
        tensor_stride=t.tensor_stride,
    )
    B = t.C[:, 0].max() + 1
    return t_pos.dense(shape=torch.Size([B, t.shape[1], res, res, res]))[0]

def sparse_clamp(input, *args, **kwargs):
    return ME.MinkowskiFunctional._wrap_tensor(input, torch.clamp(input.F, *args, **kwargs))

def bias_act_sparse(x, b=None, dim=1, act='linear', alpha=None, gain=None, clamp=None, impl='cuda', sparse_type='minkowski', res=None, force_no_mask=False):
    if sparse_type == 'pytorch':
        mask = (x == 0).all(dim=1, keepdim=True).__invert__()      # need mask because bias is added
        if force_no_mask:
            mask = torch.ones_like(mask)
        return bias_act.bias_act(x, b=b, dim=dim, act=act, alpha=alpha, gain=gain, clamp=clamp, impl=impl) * mask
    elif sparse_type == 'minkowski':
        assert clamp is None or clamp >= 0

        spec = sparse_activation_funcs[act]
        alpha = float(alpha if alpha is not None else spec.def_alpha)
        gain = float(gain if gain is not None else spec.def_gain)
        clamp = float(clamp if clamp is not None else -1)

        # Add bias.
        if b is not None:
            b = b.unsqueeze(0).repeat(x.shape[0], 1)
            b = ME.SparseTensor(
                features=b,
                coordinate_manager=x.coordinate_manager,
                coordinate_map_key=x.coordinate_map_key,
            )
            x += b

        # Evaluate activation function.
        alpha = float(alpha)
        x = spec.func(x, alpha=alpha)

        # Scale by gain.
        gain = torch.tensor(gain, dtype=float)
        if gain != 1:
            x = x * gain

        # Clamp.
        if clamp >= 0:
            x = sparse_clamp(x, -clamp, clamp)
        return x
    elif sparse_type == 'debug':
        assert res is not None
        x_dense = to_dense(x, res)
        x_py = bias_act_sparse(x_dense, b, dim, act, alpha, gain, clamp, impl, sparse_type='pytorch')
        x = bias_act_sparse(x, b, dim, act, alpha, gain, clamp, impl, sparse_type='minkowski')
        x_mink = to_dense(x, res)
        print((x_mink - x_py).abs().max())
        return x
    else:
        raise AttributeError

def spatial2coo(x):
    ''' transforms tensor volume to COO '''
    spatial_dim = x.ndim - 2
    x_coo = x.permute(0, *(2 + i for i in range(spatial_dim)), 1).reshape(-1, x.size(1))
    return x_coo

def sparse_conversion(x, final_resolution, coords=None, sparse_type='minkowski', res=None):
    ''' coords of shape [1, C, S * N]'''
    if sparse_type == 'pytorch':
        return x
    elif sparse_type == 'minkowski':
        tensor_shape = x.shape
        all_strides = [final_resolution//tensor_shape[-3], final_resolution//tensor_shape[-2], final_resolution//tensor_shape[-1]]
        if coords is None:
            coords = ME.MinkowskiOps.dense_coordinates(x.shape)
            coords = coords * torch.tensor([1, *all_strides]).int()
        x_coo = spatial2coo(x)
        x = ME.SparseTensor(x_coo.to(torch.float32), coords, tensor_stride=all_strides, device=x.device)        # need fp32 for sparse ops
        return x
    elif sparse_type == 'debug':
        assert res is not None
        x_dense = x
        x_py = sparse_conversion(x_dense, final_resolution, coords, sparse_type='pytorch')
        x = sparse_conversion(x, final_resolution, coords, sparse_type='minkowski')
        x_mink = to_dense(x, res)
        print((x_mink - x_py).abs().max())
        return x
    else:
        raise AttributeError()

def to_sparse(x: torch.Tensor, format: str = None, coordinates=None, stride=1, device=None):
    r"""Convert a batched tensor (dimension 0 is the batch dimension) to a SparseTensor
    :attr:`x` (:attr:`torch.Tensor`): a batched tensor. The first dimension is the batch dimension.
    :attr:`format` (:attr:`str`): Format of the tensor. It must include 'B' and 'C' indicating the batch and channel dimension respectively. The rest of the dimensions must be 'X'. .e.g. format="BCXX" if image data with BCHW format is used. If a 3D data with the channel at the last dimension, use format="BXXXC" indicating Batch X Height X Width X Depth X Channel. If not provided, the format will be "BCX...X".
    :attr:`device`: Device the sparse tensor will be generated on. If not provided, the device of the input tensor will be used.
    """
    assert x.ndim > 2, "Input has 0 spatial dimension."
    assert isinstance(x, torch.Tensor)
    if format is None:
        format = [
            "X",
        ] * x.ndim
        format[0] = "B"
        format[1] = "C"
        format = "".join(format)
    assert x.ndim == len(format), f"Invalid format: {format}. len(format) != x.ndim"
    assert (
        "B" in format and "B" == format[0] and format.count("B") == 1
    ), "The input must have the batch axis and the format must include 'B' indicating the batch axis."
    assert (
        "C" in format and format.count("C") == 1
    ), "The format must indicate the channel axis"
    if device is None:
        device = x.device
    ch_dim = format.find("C")
    reduced_x = torch.abs(x).sum(ch_dim)
    bcoords = torch.where(reduced_x != 0)
    stacked_bcoords = torch.stack(bcoords, dim=1).int()
    indexing = [f"bcoords[{i}]" for i in range(len(bcoords))]
    indexing.insert(ch_dim, ":")
    features = torch.zeros(
        (len(stacked_bcoords), x.size(ch_dim)), dtype=x.dtype, device=x.device
    )
    exec("features[:] = x[" + ", ".join(indexing) + "]")
    return ME.SparseTensor(
            features=features,
            coordinates=stacked_bcoords,
            tensor_stride=stride,
            device=device,
        )

def sparse_to_lists(x):
    if isinstance(x, torch.Tensor):     # dense inputs, filter by density
        coords = torch.nonzero(x[:, -1])
        _, idcs = coords[:, 0].sort()
        coords = coords[idcs]           # sort by batch

        B = coords[:, 0].max() + 1
        density_data_list = []
        sh_data_list = []
        for i in range(B):
            coords_i = coords[coords[:, 0] == i]
            d_i = x[:, -1:][coords_i[:, 0], :, coords_i[:, 1], coords_i[:, 2], coords_i[:, 3]]
            sh_i = x[:, :-1][coords_i[:, 0], :, coords_i[:, 1], coords_i[:, 2], coords_i[:, 3]]
            density_data_list.append(d_i)
            sh_data_list.append(sh_i)
        return density_data_list, sh_data_list, coords

    # split sparse tensor into SH and density
    svg = ME.SparseTensor(
        features=x.F[:, :-1],
        coordinates=x.C,
        coordinate_manager=x.coordinate_manager,
    )
    cls = ME.SparseTensor(
        features=x.F[:, -1:],
        coordinates=x.C,
        coordinate_manager=x.coordinate_manager,
    )

    # convert sparse tensor to lists
    sh_data_list = svg.decomposed_features
    density_data_list = cls.decomposed_features
    coords = svg.coordinates
    batch_tensor_stride = torch.tensor([1] + svg.tensor_stride, device=svg.device).view(1, 4)
    coords = coords.div(batch_tensor_stride, rounding_mode='floor')
    return density_data_list, sh_data_list, coords


def sparseconv3d(x, w, final_resolution, start_resolution=32, f=None, up=1, down=1, padding=0, type='minkowski', input_res=None, force_no_mask=False):
    if type == 'pytorch':
        mask = (x != 0).any(dim=1, keepdim=True).type_as(x)
        w = w.permute(0, 1, 4, 3, 2)
        if up == 2:
            mask = torch.nn.functional.upsample(mask, scale_factor=up, mode='trilinear')
            mask = (mask > 0).float()
        if force_no_mask:
            mask = torch.ones_like(mask)
        return denseconv3d(x, w, f=None, up=up, padding=padding, groups=1).to(x.dtype) * mask

    elif type == 'minkowski':
        # params
        is_transpose = (up == 2)
        stride = 2 if is_transpose else 1
        kernel_size = w.shape[-1]

        # generate kernel
        w = w.flatten(2).permute(2, 1, 0).contiguous()
        kernel_generator = KernelGenerator(
            kernel_size=kernel_size,
            stride=stride,
            dilation=1,
            expand_coordinates=is_transpose,
            dimension=3,
        )

        # params
        conv = ME.MinkowskiConvolutionTransposeFunction if is_transpose else ME.MinkowskiConvolutionFunction

        # Get a new coordinate_map_key or extract one from the coords
        out_coordinate_map_key = _get_coordinate_map_key(x, None, kernel_generator.expand_coordinates)

        outfeat = conv.apply(
            x.F.float(),
            w.float(),
            kernel_generator,
            ConvolutionMode.DEFAULT,
            x.coordinate_map_key,
            out_coordinate_map_key,
            x._manager,
        )

        outfeatsparse = ME.SparseTensor(outfeat, coordinate_map_key=out_coordinate_map_key, coordinate_manager=x._manager)

        if is_transpose:
            # kick out the max values, equivalent to cropping: out = out[..., :-1, :-1, :-1]
            _, max_coord = get_cur_min_max_coordinate(outfeatsparse, final_resolution=final_resolution, start_resolution=start_resolution)

            # prune (don't calculate on batch dimension!)
            max_idcs = torch.any(outfeatsparse.C[:, 1:] == max_coord, dim=1)
            outfeatsparse = ME.MinkowskiPruning()(outfeatsparse, ~max_idcs)

        return outfeatsparse
    elif type == 'debug':
        assert input_res is not None
        x_dense = to_dense(x, input_res)
        x_py = sparseconv3d(x_dense, w.float(), final_resolution, start_resolution, f, up, down, padding, type='pytorch', force_no_mask=force_no_mask)
        x = sparseconv3d(x, w.float(), final_resolution, start_resolution, f, up, down, padding, type='minkowski', force_no_mask=force_no_mask)
        x_mink = to_dense(x, input_res*up)
        print((x_mink-x_py).abs().max())
        return x
    else:
        raise AttributeError('type has to be "pytorch" or "minkowski".')


@misc.profiled_function
def modulated_sparseconv3d(
    x,                          # Input tensor of shape [batch_size, in_channels, in_height, in_width].
    weight,                     # Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
    styles,                     # Modulation coefficients of shape [batch_size, in_channels].
    final_resolution,
    start_resolution,
    noise           = None,     # Optional noise tensor to add to the output activations.
    up              = 1,        # Integer upsampling factor.
    down            = 1,        # Integer downsampling factor.
    padding         = 0,        # Padding with respect to the upsampled image.
    resample_filter = None,     # Low-pass filter to apply when resampling activations. Must be prepared beforehand by calling upfirdn2d.setup_filter().
    demodulate      = True,     # Apply weight demodulation?
    flip_weight     = True,     # False = convolution, True = correlation (matches torch.nn.functional.conv2d).
    fused_modconv   = True,     # Perform modulation, convolution, and demodulation as a single fused operation?
    sparse_type     = 'minkowski',   # type of sparse operations, "pytorch", "minkowski" or "debug"
    input_res       = None,     # Only needed for sparse_type == "debug"
):
    batch_size = styles.shape[0]
    out_channels, in_channels, kh, kw, kd = weight.shape
    misc.assert_shape(weight, [out_channels, in_channels, kh, kw, kd]) # [OIkk]
    misc.assert_shape(styles, [batch_size, in_channels]) # [NI]

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (1 / np.sqrt(in_channels * kh * kw * kd) / weight.norm(float('inf'), dim=[1,2,3,4], keepdim=True)) # max_Ikk
        styles = styles / styles.norm(float('inf'), dim=1, keepdim=True) # max_I

    # Calculate per-sample weights and demodulation coefficients.
    w = None
    dcoefs = None
    if demodulate or fused_modconv:
        w = weight.unsqueeze(0) # [NOIkkk]
        w = w * styles.reshape(batch_size, 1, -1, 1, 1, 1) # [NOIkkk]
    if demodulate:
        dcoefs = (w.square().sum(dim=[2,3,4,5]) + 1e-8).rsqrt() # [NO]
    if demodulate and fused_modconv:
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1, 1) # [NOIkkk]

    # Execute by scaling the activations before and after the convolution.
    if sparse_type == 'minkowski' or sparse_type == 'debug':
        styles = torch.index_select(styles, 0, x.C[:, 0])
        x = ME.SparseTensor(
            x.F * styles,
            coordinate_manager=x.coordinate_manager,
            coordinate_map_key=x.coordinate_map_key
        )
    elif sparse_type == 'pytorch':
        x = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1, 1)
    else:
        raise AttributeError

    x = sparseconv3d(x=x, w=weight.to(x.dtype), final_resolution=final_resolution, start_resolution=start_resolution, up=up, down=down, padding=padding, type=sparse_type, input_res=input_res)

    # demodulate
    if demodulate:
        if sparse_type == 'minkowski' or sparse_type == 'debug':
            # dcoefs = dcoefs.to(x.dtype).repeat(1, x.shape[0] // batch_size).reshape(-1, out_channels)  # more reliable than MinkowskiBroadcast
            dcoefs = torch.index_select(dcoefs, 0, x.C[:, 0])
            x = ME.SparseTensor(
                x.F * dcoefs,
                coordinate_manager=x.coordinate_manager,
                coordinate_map_key=x.coordinate_map_key,
            )
        elif sparse_type == 'pytorch':
            x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1, 1)
        else:
            raise AttributeError(f'Unknown sparse_type {sparse_type}')

    if noise is not None:
        x += noise

    return x


def sparse_upsample3d(
        x,  # Input tensor of shape [batch_size, in_channels, in_height, in_width, in_depth].
        target=None,
        coordinates=None,  # If provided, generate results on the provided coordinates
        upsampling_mode='nearest',  # Choose from ['nearest', 'trilinear']
        type='minkowski',
        input_res=None,
):
    if type == 'pytorch':
        if upsampling_mode == 'nearest':
            return dense_upsample3d(x, up=2, upsampling_mode=upsampling_mode)
        elif upsampling_mode == 'trilinear':     # implement as convolution due to 'cross pattern'
            # pad inputs with zeros similar to transposed convolution - careful: this only works when target was obtained with transposed convolution
            xp = x
            for i in range(1, 4):
                xp = torch.stack([torch.zeros_like(xp), xp], dim=-i).flatten(-(i + 1), -i)
            w = torch.ones([x.shape[1], 1, 3, 3, 3]).to(memory_format=torch.contiguous_format, device=x.device)
            w[:, :] = torch.tensor([
                [
                    [1,2,1],
                    [2,4,2],
                    [1,2,1],
                ],
                [
                    [2,4,2],
                    [4,8,4],
                    [2,4,2],
                ],
                [
                    [1,2,1],
                    [2,4,2],
                    [1,2,1],
                ]

            ])/8
            xup = F.conv3d(xp, w, stride=1, padding=1, groups=x.shape[1])

            if target is not None:
                assert target.shape[-1] == xup.shape[-1]
                mask = (target != 0).any(dim=1, keepdim=True)
                xup = xup * mask
            elif coordinates is not None:
                raise NotImplementedError
            return xup

        else:
            raise NotImplementedError

    elif type == 'minkowski':
        assert isinstance(x, ME.SparseTensor)
        assert x.D == 3
        assert upsampling_mode in ['nearest', 'trilinear']

        # Get a new coordinate map key or extract one from the coordinates
        if upsampling_mode == 'nearest':
            kernel_generator = KernelGenerator(kernel_size=2, stride=2, dilation=1, dimension=3)
            out_coordinate_map_key = _get_coordinate_map_key(x, coordinates, tensor_stride=[s // 2 for s in x.tensor_stride])
            outfeat = ME.MinkowskiLocalPoolingTransposeFunction.apply(
                x.F,
                PoolingMode.LOCAL_AVG_POOLING,
                kernel_generator,
                x.coordinate_map_key,
                out_coordinate_map_key,
                x._manager,
            )
            x_up = ME.SparseTensor(
                outfeat,
                coordinate_map_key=out_coordinate_map_key,
                coordinate_manager=x.coordinate_manager,
            )
        elif upsampling_mode == 'trilinear':
            if coordinates is None and target is None:
                raise AttributeError('coordinates/target have to be specified with upsampling_mode=trilinear')
            if target is not None:
                coordinates = target.C
                out_feat, _, _, _ = ME.MinkowskiInterpolationFunction.apply(
                    x.F,
                    coordinates.to(x.dtype),
                    x.coordinate_map_key,
                    x._manager,
                )
                x_up = ME.SparseTensor(
                        features=out_feat,  # Convert to a tensor
                        coordinate_manager=target.coordinate_manager,
                        coordinate_map_key=target.coordinate_map_key,
                    )
            else:
                out_feat, _, _, _ = ME.MinkowskiInterpolationFunction.apply(
                    x.F,
                    coordinates.to(x.dtype),
                    x.coordinate_map_key,
                    x._manager,
                )
                x_up = ME.SparseTensor(
                        features=out_feat,  # Convert to a tensor
                        coordinates=coordinates,
                        tensor_stride=[s//2 for s in x.tensor_stride],
                        coordinate_manager=x.coordinate_manager
                )
        else:
            raise NotImplementedError

        return x_up
    elif type == 'debug':
        assert input_res is not None
        assert input_res is not None
        x_dense = to_dense(x, input_res)
        target_dense = to_dense(target, input_res * 2)
        x_py = sparse_upsample3d(x_dense, target_dense, coordinates, upsampling_mode, type='pytorch')
        x_up = sparse_upsample3d(x, target, coordinates, upsampling_mode, type='minkowski')
        x_mink = to_dense(x_up, input_res * 2)
        print((x_mink - x_py).abs().max())
        return x_up
    else:
        raise AttributeError

#----------------------------------------------------------------------------

@persistence.persistent_class
class Conv3dLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        final_resolution,               # Networks final output (largest) resolution.
        dense_resolution,               # Networks first input (smallest) resolution
        resolution,                     # Resolution of this layer.
        kernel_size,                    # Width and height of the convolution kernel.
        bias            = True,         # Apply additive bias before the activation function?
        activation      = 'linear',     # Activation function: 'relu', 'lrelu', etc.
        up              = 1,            # Integer upsampling factor.
        down            = 1,            # Integer downsampling factor.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp      = None,         # Clamp the output to +-X, None = disable clamping.
        channels_last   = False,        # Expect the input to have memory_format=channels_last?
        trainable       = True,         # Update the weights of this layer during training?
        force_sparse    = False,        # Needed for pruning layer at dense resolution
        sparse_type     = 'minkowski'   # type of sparse operations, "pytorch", "minkowski" or "debug"
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.final_resolution = final_resolution
        self.dense_resolution = dense_resolution
        self.resolution = resolution
        self.activation = activation
        self.up = up
        self.down = down
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 3))
        self.act_gain = bias_act.activation_funcs[activation].def_gain
        self.is_sparse = resolution > dense_resolution or force_sparse
        self.sparse_type = "dense" if not self.is_sparse else sparse_type

        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size, kernel_size]).to(memory_format=memory_format)
        bias = torch.zeros([out_channels]) if bias else None
        if trainable:
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(bias) if bias is not None else None
        else:
            self.register_buffer('weight', weight)
            if bias is not None:
                self.register_buffer('bias', bias)
            else:
                self.bias = None

    def forward(self, x, gain=1, force_no_mask=False):
        w = self.weight * self.weight_gain
        b = self.bias.to(x.dtype) if self.bias is not None else None
        flip_weight = (self.up == 1) # slightly faster
        if self.is_sparse:
            x = sparseconv3d(x=x, w=w.to(x.dtype), final_resolution=self.final_resolution, start_resolution=self.dense_resolution, up=self.up, down=self.down, padding=self.padding, type=self.sparse_type, input_res=self.resolution//self.up, force_no_mask=force_no_mask)
        else:
            x = denseconv3d(x=x, w=w.to(x.dtype), f=self.resample_filter, up=self.up, down=self.down, padding=self.padding, flip_weight=flip_weight)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        if self.is_sparse:
            x = bias_act_sparse(x, b, act=self.activation, gain=act_gain, clamp=act_clamp, sparse_type=self.sparse_type, res=self.resolution, force_no_mask=force_no_mask)
        else:
            x = bias_act.bias_act(x, b, act=self.activation, gain=act_gain, clamp=act_clamp)
        return x

    def extra_repr(self):
        return ' '.join([
            f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, activation={self.activation:s},',
            f'up={self.up}, down={self.down}, is_sparse={self.is_sparse}'])

#----------------------------------------------------------------------------


class SparsePruning(ME.MinkowskiPruning):
    """Pickle-able version"""
    def __init__(self):
        super().__init__()

    def __reduce__(self):
        return (SparsePruning, ())


@persistence.persistent_class
class PruningLayer(torch.nn.Module):
    def __init__(self, resolution, dense_resolution, renderer, min_sparsity=0.3, sparse_type='minkowski'):
        super().__init__()
        if resolution < dense_resolution:
            raise AttributeError('resolution cannot be smaller than dense resolution')
        self.resolution = resolution
        self._pruning = SparsePruning()
        self.min_sparsity = min_sparsity            # keep at most (1-min_sparsity)*100 percent of the voxels when pruning with weights

        self.check_empty_neighbors = Conv3dLayer(1, 1, final_resolution=resolution, dense_resolution=dense_resolution, resolution=resolution,
                                                 kernel_size=3, bias=False, up=1, channels_last=False, trainable=False, force_sparse=True, sparse_type=sparse_type)
        self.check_empty_neighbors.weight = torch.ones_like(self.check_empty_neighbors.weight)
        self.renderer = renderer
        self.sparse_type = sparse_type

    def pruning(self, x, keep):
        if self.sparse_type == 'minkowski' or self.sparse_type == 'debug':
            x = self._pruning(x, keep)
            return x
        elif self.sparse_type == 'pytorch':
            keep = keep.view([x.shape[0], *x.shape[2:]]).unsqueeze(1)
            return x * keep
        else:
            raise AttributeError(f'Unknown sparse_type {self.sparse_type}')

    def enforce_min_sparsity(self, mask, weights):
        max_elements = round((1-self.min_sparsity) * self.resolution**3)

        # for each sample, keep only the k=max_elements voxels with the largest weights
        if self.sparse_type == 'minkowski':
            weights_list = weights.decomposed_features
        elif self.sparse_type == 'pytorch':
            weights_list = weights.flatten(2, 4).permute(0, 2, 1)
        else:
            raise AttributeError(f'Unknown sparse_type {self.sparse_type}')

        idx_start = 0
        for w in weights_list:
            w = w.squeeze(1)
            idx_end = idx_start + len(w)
            m = mask[idx_start:idx_end]

            if m.sum() > max_elements:
                w = torch.where(m, w, torch.full_like(w, fill_value=-float('inf')))
                idcs_keep = torch.topk(w, k=max_elements, sorted=False).indices
                m[:] = False
                m[idcs_keep] = True
                mask[idx_start:idx_end] = m

            idx_start = idx_end
        assert idx_end == len(mask)
        return mask

    def get_mask_for_pruning(self, weights, pose, threshold=0, keep_at_least=10):
        if self.sparse_type == 'minkowski':
            mask = torch.ones(len(weights), dtype=torch.bool, device=weights.device)  # default, keep all
            density_data_list = weights.decomposed_features
            sh_data_list = [torch.zeros_like(d).expand(-1, 3*self.renderer.basis_dim).contiguous() for d in density_data_list]
            coords = (weights.C / torch.tensor([[1]+weights.tensor_stride], device=weights.device)).long()
        elif self.sparse_type == 'pytorch':
            coords = torch.nonzero(weights[:, -1])
            _, idcs = coords[:, 0].sort()
            coords = coords[idcs]  # sort by batch

            B = coords[:, 0].max() + 1
            density_data_list = []
            sh_data_list = []
            for i in range(B):
                coords_i = coords[coords[:, 0] == i]
                d_i = weights[coords_i[:, 0], :, coords_i[:, 1], coords_i[:, 2], coords_i[:, 3]]
                sh_i = torch.ones_like(d_i).expand(-1, 3*self.renderer.basis_dim).contiguous()
                density_data_list.append(d_i)
                sh_data_list.append(sh_i)

            mask = (weights.flatten(2, 4).permute(0, 2, 1) != 0).flatten()  # default, keep all if not empty      - this follows the dense convention, whereas lists are sparse for rendering
        elif self.sparse_type == 'debug':
            x_dense = to_dense(weights, self.resolution)
            self.sparse_type = self.check_empty_neighbors.sparse_type = 'pytorch'
            mask_py = self.get_mask_for_pruning(x_dense, pose, threshold, keep_at_least)
            self.sparse_type = self.check_empty_neighbors.sparse_type = 'minkowski'
            mask = self.get_mask_for_pruning(weights, pose, threshold, keep_at_least)
            print(mask.sum(), mask_py.sum())
            self.sparse_type = self.check_empty_neighbors.sparse_type = 'debug'
            return mask
        else:
            raise AttributeError(f'Unknown sparse_type {self.sparse_type}')
        visible_voxels = self.renderer.find_visible(density_data_list, sh_data_list, coords, pose, self.resolution, self.resolution)

        # Check neighbors
        if self.sparse_type == 'minkowski':
            weights = ME.SparseTensor(
                features=weights.F * visible_voxels.int(),
                coordinate_manager=weights.coordinate_manager,
                coordinate_map_key=weights.coordinate_map_key,
            )
            is_filled = (weights.F > threshold) & mask.unsqueeze(1)
            is_filled = ME.SparseTensor(coordinates=weights.coordinates, features=is_filled.to(torch.float32), tensor_stride=weights.tensor_stride)
        elif self.sparse_type == 'pytorch':
            weights_sparse = torch.cat(density_data_list) * visible_voxels.int()
            weights = torch.zeros_like(weights)
            weights[coords[:, 0], :, coords[:, 1], coords[:, 2], coords[:, 3]] = weights_sparse
            is_filled = ((weights > threshold) & mask.view(weights.shape)).to(weights.dtype)
        else:
            raise AttributeError(f'Unknown sparse_type {self.sparse_type}')

        keep = self.check_empty_neighbors(is_filled, force_no_mask=True)        # for pytorch implementation do not mask outputs with inputs (because input is bool and zero would be interpreted as empty)

        if self.sparse_type == 'minkowski':
            if len(keep.C) != len(weights.C):
                keep = sparse_upsample3d(keep, input_res=self.resolution, target=weights, upsampling_mode='trilinear')            # TODO: find out why this is needed, seems that sometimes a coordinate just gets lost?
            keep = (keep.F > 0)

            keep = ME.SparseTensor(coordinates=weights.coordinates, features=keep.to(torch.float32), tensor_stride=weights.tensor_stride)
            if len(keep.C) != len(weights.C):
                keep = sparse_upsample3d(keep, input_res=self.resolution, target=weights, upsampling_mode='trilinear')            # TODO: find out why this is needed, seems that sometimes a coordinate just gets lost?

            keep_list = keep.decomposed_features
            weights_list = weights.decomposed_features
        elif self.sparse_type == 'pytorch':
            keep_list = (keep > 0).flatten(2, 4).permute(0, 2, 1)
            weights_list = weights.flatten(2, 4).permute(0, 2, 1)
        else:
            raise AttributeError(f'Unknown sparse_type {self.sparse_type}')

        # check for each batch sample if it has enough entries
        idx_start = 0
        for k, w in zip(keep_list, weights_list):
            w = w.squeeze(1)
            k = k.squeeze(1).bool()
            idx_end = idx_start + len(k)

            has_enough_entries = k.sum() > keep_at_least
            if not has_enough_entries:
                m = mask[idx_start:idx_end]
                w = torch.where(m, w, torch.tensor([-1e6], device=w.device,
                                                   dtype=w.dtype))  # large negative value to avoid using pruned sphere bound
                k = torch.zeros_like(k)
                idcs_keep = torch.topk(w, k=keep_at_least, sorted=False).indices
                k[idcs_keep] = 1

            mask[idx_start:idx_end] = k
            idx_start = idx_end
        assert idx_end == len(mask)

        mask = self.enforce_min_sparsity(mask, weights)

        return mask

    def forward(self, x, img, pose):
        assert img.shape[1] == 4
        if self.sparse_type == 'minkowski':
            cls = ME.SparseTensor(
                features=img.F[:, -1:],
                coordinate_manager=img.coordinate_manager,
                coordinate_map_key=img.coordinate_map_key,
            )
        elif self.sparse_type == 'pytorch':
            cls = img[:, -1:]
        elif self.sparse_type == 'debug':
            cls_py = to_dense(img, self.resolution)[:, -1:]
            cls = ME.SparseTensor(
                features=img.F[:, -1:],
                coordinate_manager=img.coordinate_manager,
                coordinate_map_key=img.coordinate_map_key,
            )
            cls_dense = to_dense(cls, self.resolution)
            print((cls_dense == cls_py).all())

            # # Pruning is not yet exactly equivalent
            # x_dense = to_dense(x, self.resolution)
            # img_dense = to_dense(img, self.resolution)
            # self.sparse_type = self.check_empty_neighbors.sparse_type = 'pytorch'
            # x_py, img_py = self(x_dense, img_dense, pose)
            # self.sparse_type = self.check_empty_neighbors.sparse_type = 'minkowski'
            # x, img = self(x, img, pose)
            # x_mink = to_dense(x, self.resolution)
            # img_mink = to_dense(img, self.resolution)
            # print((x_mink - x_py).abs().max())
            # print((img_mink - img_py).abs().max())
            # self.sparse_type = self.check_empty_neighbors.sparse_type = 'debug'
            # return x, img
        else:
            raise AttributeError(f'Unknown sparse_type {self.sparse_type}')

        keep = self.get_mask_for_pruning(cls, pose)
        if not (keep == 1).all():
            img = self.pruning(img, keep)
            x = self.pruning(x, keep)

        return x, img
#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        w_dim,                          # Intermediate latent (W) dimensionality.
        final_resolution,
        dense_resolution,
        resolution,                     # Resolution of this layer.
        kernel_size     = 3,            # Convolution kernel size.
        up              = 1,            # Integer upsampling factor.
        use_noise       = True,         # Enable noise input?
        activation      = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp      = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        channels_last   = False,        # Use channels_last format for the weights?
        sparse_type     = 'minkowski'   # type of sparse operations, "pytorch", "minkowski" or "debug"
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.final_resolution = final_resolution
        self.dense_resolution = dense_resolution
        self.resolution = resolution
        self.up = up
        self.use_noise = use_noise
        self.activation = activation
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.act_gain = bias_act.activation_funcs[activation].def_gain
        self.is_sparse = resolution > dense_resolution
        self.sparse_type = 'dense' if not self.is_sparse else sparse_type

        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size, kernel_size]).to(memory_format=memory_format))
        if use_noise:
            self.register_buffer('noise_const', torch.randn([resolution, resolution, resolution]))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

    def forward(self, x, w, noise_mode='random', fused_modconv=True, gain=1):
        assert noise_mode in ['random', 'const', 'none']
        if not self.is_sparse:
            in_resolution = self.resolution // self.up
            misc.assert_shape(x, [None, self.in_channels, in_resolution, in_resolution, in_resolution])
        styles = self.affine(w)

        noise = None
        if noise_mode != 'none':
            if self.is_sparse:
                if noise_mode == 'random':
                    noise = torch.randn(len(x.F), device=x.device) * self.noise_strength
                if noise_mode == 'const':
                    coord_idx = (x.C / torch.tensor([1] + x.tensor_stride, device=x.device)).long()
                    noise = self.noise_const[coord_idx[:, 1], coord_idx[:, 2], coord_idx[:, 3]] * self.noise_strength

                noise = ME.SparseTensor(
                    features=noise.unsqueeze(1).expand(-1, self.out_channels),
                    coordinate_manager=x.coordinate_manager,
                    coordinate_map_key=x.coordinate_map_key,
                )
            else:
                if self.use_noise and noise_mode == 'random':
                    noise = torch.randn([x.shape[0], 1, self.resolution, self.resolution, self.resolution], device=x.device) * self.noise_strength
                if self.use_noise and noise_mode == 'const':
                    noise = self.noise_const * self.noise_strength

        flip_weight = (self.up == 1) # slightly faster
        if self.is_sparse:
            #TODO: fix noise issue for upsampling
            x = modulated_sparseconv3d(x=x, weight=self.weight, styles=styles, noise=noise, final_resolution=self.final_resolution, start_resolution=self.dense_resolution,
                                       up=self.up, padding=self.padding, resample_filter=self.resample_filter, flip_weight=flip_weight, fused_modconv=fused_modconv, sparse_type=self.sparse_type, input_res=self.resolution//self.up)

        else:
            x = modulated_denseconv3d(x=x, weight=self.weight, styles=styles, noise=noise, up=self.up,
                padding=self.padding, resample_filter=self.resample_filter, flip_weight=flip_weight, fused_modconv=fused_modconv)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None

        if self.is_sparse:
            x = bias_act_sparse(x, self.bias.to(x.dtype), act=self.activation, gain=act_gain, clamp=act_clamp, sparse_type=self.sparse_type, res=self.resolution)
        else:
            x = bias_act.bias_act(x, self.bias.to(x.dtype), act=self.activation, gain=act_gain, clamp=act_clamp)
        return x

    def extra_repr(self):
        return ' '.join([
            f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, w_dim={self.w_dim:d},',
            f'resolution={self.resolution:d}, up={self.up}, activation={self.activation:s}'])

#----------------------------------------------------------------------------

@persistence.persistent_class
class ToRGBLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        w_dim,
        final_resolution,
        dense_resolution,
        resolution,
        kernel_size=1,
        conv_clamp=None,
        channels_last=False,
        sparse_type='minkowski'
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.final_resolution = final_resolution
        self.dense_resolution = dense_resolution
        self.resolution = resolution
        self.conv_clamp = conv_clamp
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size, kernel_size]).to(memory_format=memory_format))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 3))
        self.is_sparse = resolution > dense_resolution
        self.sparse_type = 'dense' if not self.is_sparse else sparse_type

    def forward(self, x, w, fused_modconv=True):
        styles = self.affine(w) * self.weight_gain
        if self.is_sparse:
            x = modulated_sparseconv3d(x=x, weight=self.weight, styles=styles, final_resolution=self.final_resolution, start_resolution=self.dense_resolution,
                                       demodulate=False, fused_modconv=fused_modconv, sparse_type=self.sparse_type, input_res=self.resolution)
            x = bias_act_sparse(x, self.bias.to(x.dtype), clamp=self.conv_clamp, sparse_type=self.sparse_type, res=self.resolution)
        else:
            x = modulated_denseconv3d(x=x, weight=self.weight, styles=styles, demodulate=False, fused_modconv=fused_modconv)
            x = bias_act.bias_act(x, self.bias.to(x.dtype), clamp=self.conv_clamp)
        return x

    def extra_repr(self):
        return f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, w_dim={self.w_dim:d}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                            # Number of input channels, 0 = first block.
        out_channels,                           # Number of output channels.
        w_dim,                                  # Intermediate latent (W) dimensionality.
        final_resolution,
        dense_resolution,
        resolution,                             # Resolution of this block.
        img_channels,                           # Number of output color channels.
        is_last,                                # Is this the last block?
        architecture            = 'skip',       # Architecture: 'orig', 'skip', 'resnet'.
        resample_filter         = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp              = 256,          # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16                = False,        # Use FP16 for this block?
        fp16_channels_last      = False,        # Use channels-last memory format with FP16?
        fused_modconv_default   = False,        # Default value of fused_modconv. 'inference_only' = True for inference, False for training.
        renderer                = None,         # Needed for pruning
        sparse_type             = 'minkowski',  # Type of sparse operations, "pytorch", "minkowski" or "debug"
        **layer_kwargs,                         # Arguments for SynthesisLayer.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.final_resolution = final_resolution
        self.dense_resolution = dense_resolution
        self.resolution = resolution
        self.img_channels = img_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.fused_modconv_default = fused_modconv_default
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.num_conv = 0
        self.num_torgb = 0
        self.is_sparse = resolution > dense_resolution
        self.sparse_conversion = self.resolution == self.dense_resolution  # If true, this block converts dense to sparse coordinates before returning them
        self.sparse_type = 'dense' if not (self.is_sparse or self.sparse_conversion) else sparse_type
        if self.sparse_conversion:
            self.init_coords(batch_size=1)

        if in_channels == 0:
            self.const = torch.nn.Parameter(torch.randn([out_channels, resolution, resolution, resolution]))

        if in_channels != 0:
            self.conv0 = SynthesisLayer(in_channels, out_channels, w_dim=w_dim, final_resolution=final_resolution,
                                        dense_resolution=dense_resolution,
                                        resolution=resolution, up=2, resample_filter=resample_filter,
                                        conv_clamp=conv_clamp, channels_last=self.channels_last, sparse_type=sparse_type, **layer_kwargs)
            self.num_conv += 1

        self.conv1 = SynthesisLayer(out_channels, out_channels, w_dim=w_dim, final_resolution=final_resolution,
                                    dense_resolution=dense_resolution,
                                    resolution=resolution, conv_clamp=conv_clamp, channels_last=self.channels_last, sparse_type=sparse_type,
                                    **layer_kwargs)
        self.num_conv += 1

        if is_last or architecture == 'skip':
            self.torgb = ToRGBLayer(out_channels, img_channels, w_dim=w_dim, final_resolution=final_resolution,
                                    dense_resolution=dense_resolution, resolution=resolution,
                                    conv_clamp=conv_clamp, channels_last=self.channels_last, sparse_type=sparse_type)
            self.num_torgb += 1

        if in_channels != 0 and architecture == 'resnet':
            self.skip = Conv3dLayer(in_channels, out_channels, final_resolution=final_resolution, kernel_size=3,
                                    dense_resolution=dense_resolution, resolution=resolution,
                                    bias=False, up=2, resample_filter=resample_filter,
                                    channels_last=self.channels_last, sparse_type=sparse_type)

        if self.is_sparse or self.sparse_conversion:
            self.pruning = PruningLayer(resolution, renderer=renderer, dense_resolution=dense_resolution, sparse_type=sparse_type)

    def init_coords(self, batch_size):
        ''' cache coordinates for faster sparse conversion.'''
        if self.resolution != self.dense_resolution:
            raise AttributeError('sparse conversion should only be used when transitioning from dense to sparse')

        if self.sparse_type == 'minkowski' or self.sparse_type == 'debug':
            all_strides = 3 * [self.final_resolution // self.dense_resolution]
            self.feat_coords = ME.MinkowskiOps.dense_coordinates((batch_size, self.out_channels, *[self.dense_resolution] * 3))
            self.img_coords = ME.MinkowskiOps.dense_coordinates((batch_size, self.img_channels, *[self.dense_resolution] * 3))
            self.feat_coords *= torch.tensor([1, *all_strides]).int()
            self.img_coords *= torch.tensor([1, *all_strides]).int()
        elif self.sparse_type == 'pytorch':
            self.feat_coords = self.img_coords = None
        else:
            raise AttributeError(f'Unknown sparse_type {self.sparse_type}')

        self.curr_batch_size = batch_size

    def forward(self, x, img, ws, force_fp32=False, fused_modconv=None, update_emas=False, prune=False, pose=None, **layer_kwargs):
        _ = update_emas # unused
        misc.assert_shape(ws, [None, self.num_conv + self.num_torgb, self.w_dim])
        w_iter = iter(ws.unbind(dim=1))
        if ws.device.type != 'cuda':
            force_fp32 = True
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format
        if fused_modconv is None:
            fused_modconv = self.fused_modconv_default
        if fused_modconv == 'inference_only':
            fused_modconv = (not self.training)

        # Input.
        if self.in_channels == 0:
            x = self.const.to(dtype=dtype, memory_format=memory_format)
            x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1, 1])
            if self.is_sparse:
                x = sparse_conversion(x, self.final_resolution, sparse_type=self.sparse_type, res=self.resolution)          # caching coordinates not needed
        else:
            # misc.assert_shape(x, [None, self.in_channels, self.resolution // 2, self.resolution // 2])
            if not self.is_sparse:
                x = x.to(dtype=dtype, memory_format=memory_format)

        # Main layers.
        if self.in_channels == 0:
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
        elif self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
            if self.is_sparse:
                x = ME.SparseTensor(features=x.F, coordinate_map_key=y.coordinate_map_key, coordinate_manager=y.coordinate_manager)
            x += y
        else:
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)

        # ToRGB.
        if img is not None:
            # misc.assert_shape(img, [None, self.img_channels, self.resolution // 2, self.resolution // 2])
            if self.is_sparse:
                if not (self.is_last or self.architecture == 'skip'):        # upsample later with target for is_last or skip
                    img = sparse_upsample3d(img, upsampling_mode='trilinear', type=self.sparse_type, input_res=self.resolution//2)
            else:
                img = dense_upsample3d(img, self.resample_filter)
        if self.is_last or self.architecture == 'skip':
            y = self.torgb(x, next(w_iter), fused_modconv=fused_modconv)
            if not self.is_sparse:
                y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            if self.is_sparse:
                img = sparse_upsample3d(img, target=y, upsampling_mode='trilinear', type=self.sparse_type, input_res=self.resolution//2)         # upsample with target
            if img is not None:
                img += y
            else:
                img = y

        if self.sparse_conversion:  # Convert dense to sparse coordinates
            batch_size = ws.shape[0]
            if batch_size != self.curr_batch_size:
                self.init_coords(batch_size)
            x = sparse_conversion(x, self.final_resolution, coords=self.feat_coords, sparse_type=self.sparse_type, res=self.resolution)
            if img is not None:
                img = sparse_conversion(img, self.final_resolution, coords=self.img_coords, sparse_type=self.sparse_type, res=self.resolution)

        # Pruning
        if prune:
            assert self.is_sparse or self.sparse_conversion, "cannot prune dense outputs"
            assert (img is not None) and (pose is not None) and (self.pruning.renderer is not None)
            x, img = self.pruning(x, img, pose=pose)

        if not self.is_sparse and not self.sparse_conversion:
            assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32
        return x, img

    def extra_repr(self):
        return f'resolution={self.resolution:d}, architecture={self.architecture:s}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisNetwork(torch.nn.Module):
    def __init__(self,
        w_dim,                      # Intermediate latent (W) dimensionality.
        grid_resolution,            # Output grid resolution.
        img_channels,               # Number of color channels.
        dense_resolution,           # Use sparse operations for all higher resolutions
        channel_base    = 32768,    # Overall multiplier for the number of channels.
        channel_max     = 512,      # Maximum number of channels in any layer.
        channel_min     = 16,       # Minimum number of channels in any layer.
        num_fp16_res    = 4,        # Use FP16 for the N highest resolutions.
        prune_last = False,         # If true, prune the last layer (should only be used when resuming from model trained at that resolution).
        **block_kwargs,             # Arguments for SynthesisBlock.
    ):
        assert grid_resolution >= 4 and grid_resolution & (grid_resolution - 1) == 0
        # set global dense resolution (only needed for sparse_type == "debug")
        global __DENSE_RESOLUTION__, __SPARSE_RESOLUTION__
        __DENSE_RESOLUTION__ = dense_resolution
        __SPARSE_RESOLUTION__ = grid_resolution
        super().__init__()
        self.w_dim = w_dim
        self.grid_resolution = grid_resolution
        self.grid_resolution_log2 = int(np.log2(grid_resolution))
        self.img_channels = img_channels
        self.dense_resolution = dense_resolution
        self.num_fp16_res = num_fp16_res
        self.block_resolutions = [2 ** i for i in range(2, self.grid_resolution_log2 + 1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        channels_dict = {k: max(channel_min, v) for k, v in channels_dict.items()}
        fp16_resolution = max(2 ** (self.grid_resolution_log2 + 1 - num_fp16_res), 8)

        self.pruned_resolutions = []
        for res in self.block_resolutions:
            if res >= dense_resolution and (res != self.grid_resolution or prune_last):
                self.pruned_resolutions.append(res)

        self.num_ws = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.grid_resolution)
            block = SynthesisBlock(in_channels, out_channels, w_dim=w_dim, final_resolution=grid_resolution, dense_resolution=dense_resolution,      # start resolution of sparse inputs
                resolution=res, img_channels=img_channels, is_last=is_last, use_fp16=use_fp16, **block_kwargs)
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)

    def forward(self, ws, **block_kwargs):
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        x = img = None
        for res, cur_ws in zip(self.block_resolutions, block_ws):
            block = getattr(self, f'b{res}')
            x, img = block(x, img, cur_ws, prune=res in self.pruned_resolutions, **block_kwargs)
        return sparse_to_lists(img)

    def extra_repr(self):
        return ' '.join([
            f'w_dim={self.w_dim:d}, num_ws={self.num_ws:d},',
            f'grid_resolution={self.grid_resolution:d}, img_channels={self.img_channels:d},',
            f'num_fp16_res={self.num_fp16_res:d}'])
