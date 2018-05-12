import glob
import torch
from os import path as osp
from torch.utils.ffi import create_extension

abs_path = osp.dirname(osp.realpath(__file__))
extra_objects = [osp.join(abs_path, 'build/binop_cuda_kernel.so')]
extra_objects += glob.glob('/usr/local/cuda/lib64/*.a')

ffi = create_extension(
    'binop',
    headers=['include/binop_cuda.h']+['include/binop.h'],
    sources=['src/binop_cuda.c']+['src/binop.c'],
    define_macros=[('WITH_CUDA', None)],
    relative_to=__file__,
    with_cuda=True,
    extra_objects=extra_objects,
    include_dirs=[osp.join(abs_path, 'include')],
    extra_compile_args=["-std=c99", "-O3", "-fopenmp", "-mtune=native", "-march=x86-64"]
)


if __name__ == '__main__':
    assert torch.cuda.is_available(), 'Please install CUDA for GPU support.'
    ffi.build()
