#!/usr/bin/env python

import os
from setuptools import setup, Extension
import sys

on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
# read the docs could not compile numpy and c extensions
if on_rtd:
    extensions = []
    cmdclasses = {}
else:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
    import numpy
    import numpy.distutils
    if sys.platform == 'darwin':
        print("Platform Detection: Mac OS X. Link to openblas...")
        extra_link_args = ['-L/usr/local/opt/openblas/lib -lopenblas']
        include_dirs = (numpy.distutils.misc_util.get_numpy_include_dirs() +
                        ['/usr/local/opt/openblas/include'])
    else:
        # assume linux otherwise, unless we support Windows in the future...
        print("Platform Detection: Linux. Link to liblapacke...")
        extra_link_args = ['-llapacke -llapack -lblas']
        include_dirs = (numpy.distutils.misc_util.get_numpy_include_dirs() +
                        ['/usr/include/lapacke'])

    extensions = cythonize([
        Extension(
            "libact.query_strategies._variance_reduction",
            ["libact/query_strategies/src/variance_reduction/variance_reduction.c"],
            extra_link_args=extra_link_args,
            extra_compile_args=['-std=c11'],
            include_dirs=include_dirs,
        ),
        Extension(
            "libact.query_strategies._hintsvm",
            sources=["libact/query_strategies/_hintsvm.pyx",
                     "libact/query_strategies/src/hintsvm/libsvm_helper.c",
                     "libact/query_strategies/src/hintsvm/svm.cpp"],
            include_dirs=[numpy.get_include(),
                          "libact/query_strategies/src/hintsvm/"],
            extra_compile_args=['-lstdc++'],
        ),
    ])
    cmdclasses = {'build_ext': build_ext}


setup(
    name='libact',
    version='0.1.3b0',
    description='Pool-based active learning in Python',
    long_description=open('README.md').read(),
    author='Y.-Y. Yang, S.-C. Lee, Y.-A. Chung, T.-E. Wu, H.-T. Lin',
    author_email='b01902066@csie.ntu.edu.tw, b01902010@csie.ntu.edu.tw, '
        'b01902040@csie.ntu.edu.tw, r00942129@ntu.edu.tw, htlin@csie.ntu.edu.tw',
    url='https://github.com/ntucllab/libact',
    cmdclass=cmdclasses,
    classifiers=[
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
    ],
    test_suite='libact',
    packages=[
        'libact',
        'libact.base',
        'libact.models',
        'libact.models.multilabel',
        'libact.labelers',
        'libact.query_strategies',
        'libact.query_strategies.multilabel',
        'libact.query_strategies.multiclass',
        'libact.utils',
    ],
    package_dir={
        'libact': 'libact',
        'libact.base': 'libact/base',
        'libact.models': 'libact/models',
        'libact.labelers': 'libact/labelers',
        'libact.query_strategies': 'libact/query_strategies',
        'libact.query_strategies.multiclass': 'libact/query_strategies/multiclass',
        'libact.utils': 'libact/utils',
    },
    ext_modules=extensions,
)
