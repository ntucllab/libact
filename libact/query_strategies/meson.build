py_src = [
  '__init__.py',
  'active_learning_by_learning.py',
  'density_weighted_meta.py',
  'density_weighted_uncertainty_sampling.py',
  'hintsvm.py',
  'query_by_committee.py',
  'quire.py',
  'random_sampling.py',
  'uncertainty_sampling.py',
  'variance_reduction.py',
]

py.install_sources(
  py_src,
  subdir: 'libact/query_strategies',
)

incdir_numpy = run_command(
  py,
  [
    '-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())',
  ],
  check: true,
).stdout().strip()

blas_candidates = ['mkl', 'Accelerate', 'openblas', 'blis', 'lapack']
blas_opt = get_option('blas')
lapack_opt = get_option('lapack')

blas_dep = (blas_opt != 'auto') ? dependency(blas_opt, required: false) : dependency('', required: false)
lapack_dep = (lapack_opt != 'auto') ? dependency(lapack_opt, required: false) : dependency('', required: false)

# fall-back loop

foreach cand : blas_candidates
  if not blas_dep.found()
    blas_dep = dependency(cand, required: false)
  endif
  if not lapack_dep.found()
    lapack_dep = dependency(cand, required: false)
  endif
endforeach

if not blas_dep.found()
  error(
    'No usable BLAS library found. Ensure that necessary packages are installed and included the PATH environment variable.',
  )
endif

if not lapack_dep.found()
  error(
    'No usable LAPACK library found. Ensure that necessary packages are installed and included the PATH environment variable.',
  )
endif

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
    subdir: 'libact/query_strategies',
  )
endif

srcs = [
  '_hintsvm.pyx',
  'src/hintsvm/libsvm_helper.c',
  'src/hintsvm/svm.cpp',
]

if get_option('hintsvm')
  py.extension_module(
    '_hintsvm',
    srcs,
    dependencies: [blas_dep, lapack_dep],
    include_directories: [
      include_directories('src/hintsvm'),
      include_directories(incdir_numpy),
    ],
    install: true,
    subdir: 'libact/query_strategies',
  )
endif

subdir('multilabel')
subdir('multiclass')
subdir('tests')
