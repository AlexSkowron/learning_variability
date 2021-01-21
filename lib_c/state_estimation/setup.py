#!/Users/csmfindling/anaconda/bin/ python

from distutils.extension import Extension
from Cython.Distutils import build_ext
from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("smc_c",
                             sources=["smc_py.pyx", "smc_functions.cpp"],language='c++', extra_compile_args=["-stdlib=libc++"], extra_link_args=["-stdlib=libc++"],
                             include_dirs=[numpy.get_include(), "/usr/local/boost_1_59_0"])],
)
