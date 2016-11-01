from numpy.distutils.core import setup, Extension
from Cython.Build import cythonize

ext = Extension(
    "svm_wrap",
    sources=["svm_wrap.pyx", "svm.cpp"],
    extra_compile_args=["-std=c++11", "-O3"]
)
setup(
    name="svm_wrap",
    ext_modules = cythonize([ext])
)
