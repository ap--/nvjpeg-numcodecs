from Cython.Build import cythonize
from setuptools import setup
from setuptools.extension import Extension

setup(
    name="nvjpeg_numcodecs",
    ext_modules=cythonize(
        Extension(
            "nvjpeg_numcodecs._nvjpeg",
            sources=["nvjpeg_numcodecs/_nvjpeg.pyx"],
            libraries=["cudart", "nvjpeg"],
        ),
        compiler_directives={
            "language_level": 3,
        },
    ),
)
