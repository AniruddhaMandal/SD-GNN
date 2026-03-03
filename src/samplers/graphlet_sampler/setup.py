from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import subprocess
import os

include_dir = os.path.join(os.path.dirname(__file__), 'include')

# Detect CUDA major version to select supported architectures
def cuda_major():
    try:
        out = subprocess.check_output(['nvcc', '--version'], stderr=subprocess.STDOUT).decode()
        for token in out.split():
            if token.startswith('V'):
                return int(token[1:].split('.')[0])
    except Exception:
        return 12  # safe fallback
    return 12

cuda_ver = cuda_major()

nvcc_flags = ['-O3', '--use_fast_math']

# sm_70 (Volta) was dropped in CUDA 13
if cuda_ver < 13:
    nvcc_flags.append('-gencode=arch=compute_70,code=sm_70')

nvcc_flags += [
    '-gencode=arch=compute_80,code=sm_80',   # Ampere (A100)
    '-gencode=arch=compute_86,code=sm_86',   # Ampere (RTX 3xxx)
    '-gencode=arch=compute_89,code=sm_89',   # Ada Lovelace (RTX 4xxx)
]

# sm_90 (Hopper) supported from CUDA 11.8+, common on CUDA 13.x machines
if cuda_ver >= 12:
    nvcc_flags.append('-gencode=arch=compute_90,code=sm_90')

setup(
    name='graphlet_sampler',
    version='1.0.0',
    author='Aniruddha Mandal',
    description='GPU-parallel seed-expansion (Lifting) k-graphlet sampler with log-probability output',
    ext_modules=[
        CUDAExtension(
            name='graphlet_sampler',
            sources=[
                'src/graphlet_sampler.cpp',
                'src/graphlet_kernel.cu',
            ],
            include_dirs=[include_dir],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17', '-fopenmp'],
                'nvcc': nvcc_flags,
            },
            extra_link_args=['-fopenmp'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension,
    },
)
