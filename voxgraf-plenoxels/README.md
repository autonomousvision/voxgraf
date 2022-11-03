We adapted the CUDA kernels from [Plenoxels: Radiance Fields without Neural Networks](https://arxiv.org/abs/2112.05131).
Try installing pre-compiled CUDA extension:
```commandline
pip install ../dist/svox2-voxgraf-0.0.1.dev0+sphtexcub.lincolor.fast-cp39-cp39-linux_x86_64.whl
```
Or install from this directory by running
```commandline
pip install .
```
For more details please refer to [the Plenoxels github repository](https://github.com/sxyu/svox2).

Modifications contain: </br>
* rendering alpha
* rendering depth
* rendering variance of depth
