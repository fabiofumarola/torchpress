The Long way of Deep Learning with Torch: part 9
============
**Abstract:** In this post we analyze how to use **cuda** and **opencl** to speedup neural networks training.


Torch provides the libraries to interact with:

- Nvidia Cuda: [cutorch](https://github.com/torch/cutorch) and [cunn](https://github.com/torch/cunn)

- OpenCL: [cltorch](https://github.com/hughperkins/cltorch) and [clnn](https://github.com/hughperkins/clnn)

These libraries implement respectively torch and nn in Cuda and OpenCL. Let us analyze some of the base methods to interact with these libraries.

## Cutorch and cltorch

### General Information about GPU

To install the libraries:

```bash

# for nvidia GPU-based
luarocks install cutorch
luarocks install cunn

# for openCL GPU-based
luarocks install cltorch
luarocks install clnn

```

To get information about the system GPUs

```lua

-- get the count
cltorch.getDeviceCount()
cutorch.getDeviceCount()

-- get device properties
cltorch.getDeviceProperties(2)
cltorch.getDeviceProperties(1)

```

### Tensor operations

To transfer a tensor to the GPU

```lua

data = torch.randn(100,100)
-- select the instruction based on your GPU
dest = data:cl()
dest = data:cuda()
```

To copy a tensor from GPU to cpu

```lua

src = dest:clone()

```
This operation can be also used to clone a tensor to another GPU

```lua
cltorch.setDevice(other_device_id)
cutorch.setDevice(other_device_id)
src = dest:clone()
```
