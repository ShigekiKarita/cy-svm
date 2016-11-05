from subprocess import call
from glob import glob

from numpy.distutils.core import setup, Extension, Command
from distutils.command.clean import clean
from distutils.command.build import build
from Cython.Build import cythonize


def python(file):
    lib = glob("./build/lib*")[0]
    cmd = "PYTHONPATH=%s python %s" % (lib, file)
    print(cmd)
    call(cmd, shell=True)


class Custom(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass


class Test(Custom, build):
    def run(self):
        build.run(self)
        python("test.py")


class Benchmark(Custom, build):
    def run(self):
        build.run(self)
        python("benchmark.py")


class Clean(clean):
    def run(self):
        clean.run(self)
        if self.all:
            call("rm *_wrap.cpp", shell=True)
            call("rm -r build", shell=True)


ext = Extension(
    "svm_wrap",
    sources=["svm_wrap.pyx", "svm.cpp"],
    extra_compile_args=["-std=c++11", "-O3", "-march=native"]
)
setup(
    name="svm_wrap",
    version="1.0.0",
    ext_modules=cythonize([ext]),
    cmdclass={'clean': Clean, "test": Test, "benchmark": Benchmark}
)
