from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
from torch.utils.cpp_extension import CUDAExtension

setup(
    name='SigmoidFocalLoss',
    version="2.2.0",
    author="Chen ZHao",
    author_email="xu.frost@gmail.com",
    description="A small package for 1d aligment in cuda",
    long_description="I will write a longer description here :)",
    long_description_content_type="text/markdown",
    url="https://github.com/Frostinassiky/G-TAD",
    ext_modules=[
        CUDAExtension(
            name = 'SigmoidFocalLoss',
            sources = [
                'SigmoidFocalLoss.cpp',
                'SigmoidFocalLoss_cuda.cu',
            ],
            define_macros=[("WITH_CUDA", None)],
            extra_compile_args={'cxx': [],
              'nvcc': ['--expt-relaxed-constexpr']}
         )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
