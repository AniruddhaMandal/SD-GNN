from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

include_dir = os.path.join(os.path.dirname(__file__), 'include')

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
                'nvcc': [
                    '-O3', '--use_fast_math',
                    '-gencode=arch=compute_70,code=sm_70',
                    '-gencode=arch=compute_80,code=sm_80',
                    '-gencode=arch=compute_86,code=sm_86',
                    '-gencode=arch=compute_89,code=sm_89',
                ],
            },
            extra_link_args=['-fopenmp'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension,
    },
)
