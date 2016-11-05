from subprocess import call
from glob import glob

from numpy.distutils.core import setup, Extension, Command
from distutils.command.clean import clean
from Cython.Build import cythonize



class MyCommand(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def call(self, cmd):
        lib = glob("./build/lib*")[0]
        pre = "PYTHONPATH=%s " % lib
        print(cmd)
        call(pre + cmd, shell=True)


class Test(MyCommand):
    def run(self):
        self.call("python test.py")


class Benchmark(MyCommand):
    def run(self):
        self.call("python benchmark.py")


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
    ext_modules=cythonize([ext]),
    cmdclass={'clean': Clean, "test": Test, "benchmark": Benchmark}
)
