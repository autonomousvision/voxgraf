from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

extension = CUDAExtension('stylegan3_cuda',
    sources=[os.path.join('./ops', x) for x in ['filtered_lrelu.cpp', 'filtered_lrelu_wr.cu', 'filtered_lrelu_rd.cu',
        'filtered_lrelu_ns.cu', 'bias_act.cpp', 'bias_act_kernel.cu', 'pybind.cpp', 'upfirdn2d.cpp', 'upfirdn2d_kernel.cu']],
    headers=[os.path.join('./ops', x) for x in ['filtered_lrelu.h', 'filtered_lrelu.cu', 'bias_act.h', 'upfirdn2d.h']],
    extra_compile_args = {'cxx': [], 'nvcc': ['--use_fast_math']}
)

setup(
    name='stylegan3_cuda',
    ext_modules=[extension],
    cmdclass={
        'build_ext': BuildExtension
    })
