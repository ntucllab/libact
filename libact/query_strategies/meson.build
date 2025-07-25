if get_option('variance_reduction')
  message('Building VarianceReduction ...')
  py.extension_module(
    '_variance_reduction',
    ['src/variance_reduction/variance_reduction.c'],
    c_args: ['-std=c11'],
    dependencies: [blas_dep, lapack_dep],
    include_directories: [
      include_directories(incdir_numpy),
    ],
    install: true,
  )
endif

if get_option('hintsvm')
  message('Building HINTSVM ...')
  srcs = [
    '_hintsvm.pyx',
    'src/hintsvm/libsvm_helper.c',
    'src/hintsvm/svm.cpp',
  ]

  py.extension_module(
    '_hintsvm',
    srcs,
    dependencies: [blas_dep, lapack_dep],
    include_directories: [
      include_directories('src/hintsvm'),
      include_directories(incdir_numpy),
    ],
    override_options: ['cython_language=cpp'],
    install: true,
  )
endif
