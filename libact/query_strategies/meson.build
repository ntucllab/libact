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

# BLAS/LAPACK are only needed if variance_reduction or hintsvm features are enabled
# By default, both features are enabled (see meson.options)
want_variance_reduction = get_option('variance_reduction')
want_hintsvm = get_option('hintsvm')
need_blas = want_variance_reduction or want_hintsvm

# Show what we're attempting to build
message('Optional features requested:')
message('  - variance_reduction: ' + want_variance_reduction.to_string())
message('  - hintsvm: ' + want_hintsvm.to_string())

blas_dep = dependency('', required: false)
lapack_dep = dependency('', required: false)

if need_blas
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

  if not blas_dep.found() or not lapack_dep.found()
    # If BLAS/LAPACK not found, automatically disable optional features and continue
    warning(
      'BLAS/LAPACK libraries not found. Disabling variance_reduction and hintsvm features. ' +
      'To enable these features, install BLAS/LAPACK libraries (e.g., libopenblas-dev and liblapacke-dev on Debian/Ubuntu, openblas on macOS via Homebrew).'
    )
    want_variance_reduction = false
    want_hintsvm = false
  endif
endif

if want_variance_reduction
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
else
  message('Skipping VarianceReduction (BLAS/LAPACK not available or feature disabled)')
endif

srcs = [
  '_hintsvm.pyx',
  'src/hintsvm/libsvm_helper.c',
  'src/hintsvm/svm.cpp',
]

if want_hintsvm
  message('Building HintSVM ...')
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
else
  message('Skipping HintSVM (BLAS/LAPACK not available or feature disabled)')
endif

subdir('multilabel')
subdir('multiclass')
subdir('tests')

# Summary of what was built
message('=== Build Summary ===')
if want_variance_reduction
  message('  ✓ variance_reduction: BUILT')
else
  message('  ✗ variance_reduction: SKIPPED')
endif
if want_hintsvm
  message('  ✓ hintsvm: BUILT')
else
  message('  ✗ hintsvm: SKIPPED')
endif
message('====================')
