import glob
import torch
from os import path as osp
from torch.utils.ffi import create_extension

abs_path = osp.dirname(osp.realpath(__file__))
extra_objects = []
sources = ['src/binop.c']
headers = ['include/binop.h']
defines = []
with_cuda = False

if torch.cuda.is_available():
    extra_objects += [osp.join(abs_path, 'build/binop_cuda_kernel.so')]
    extra_objects += glob.glob('/usr/local/cuda/lib64/*.a')
    sources += ['src/binop_cuda.c']
    headers += ['include/binop_cuda.h']
    defines += [('WITH_CUDA', None)]
    with_cuda = True
ffi = create_extension(
    'binop',
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    extra_objects=extra_objects,
    include_dirs=[osp.join(abs_path, 'include')],
    extra_compile_args=["-std=c99", "-Ofast", "-fopenmp", "-mtune=native", "-march=x86-64"]
)

if __name__ == '__main__':
    ffi.build()
