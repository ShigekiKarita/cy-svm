from numpy.distutils.core import setup, Extension
from Cython.Build import cythonize

# ext = Extension("ddot_cython",
#                 sources=["ddot_cython.pyx", "ddot_c.c"])
#
# setup(
#     name="ddot_cython",
#     ext_modules = cythonize([ext])
# )


ext = Extension("svm_wrap", sources=["svm_wrap.pyx", "svm.cpp"], extra_compile_args=["-std=c++11"])
setup(name="svm_wrap", ext_modules = cythonize([ext]))