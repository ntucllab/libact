project(
  'libact',
  'c',
  'cython',
  'cpp',
  meson_version: '>=1.5.0',
  default_options: ['buildtype=release', 'c_std=c11', 'cpp_std=c++17'],
)
py = import('python').find_installation(pure: false)

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

subdir('libact/query_strategies')
